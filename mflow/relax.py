# -*- coding: utf-8 -*-
"""relax: ASE optimisers, one structure per worker process."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:                                   # peak memory reporting (unix only)
    from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage
except ImportError:                    # pragma: no cover
    getrusage = lambda _: type("rusage", (), {"ru_maxrss": 0})()
    RUSAGE_SELF = RUSAGE_CHILDREN = 0

import numpy as np
import ase.io
from ase import Atoms
from rich.progress import Progress

from . import __version__
from .calc import _hist
from .data import fix_prefix, free_prefix, group_of, read_dataset
from .log import LOG, MONITOR, elapsed, num, progress_columns, table
from .model import build_calculator, pick_device, resolve_model, versions
from .style import COLORS, LINESTYLES, MARKERS, plt


OPTIMIZERS = {"fire": "FIRE", "fire2": "FIRE2", "lbfgs": "LBFGS", "bfgs": "BFGS"}

_W: dict = {}  # per-worker state: the calculator is built once and reused

def _relax_init(spec, device, dtype, head, threads, settings):
    """Runs once per worker process — loading the model per structure would dominate."""
    import torch
    torch.set_num_threads(max(1, threads))
    _W.update(settings)
    _W["calc"] = build_calculator(spec, device, dtype, head)


def _relax_one(payload):
    """One structure, start to finish. Never raises: failures come back as a message."""
    index, numbers, positions, cell, pbc = payload
    t0 = time.perf_counter()
    result = {"index": index, "error": None, "cell_relaxed": False, "gpu_mb": 0.0}
    try:
        from ase.optimize import BFGS, FIRE, FIRE2, LBFGS
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        atoms.calc = _W["calc"]
        energy0 = float(atoms.get_potential_energy())
        volume0 = atoms.get_volume() if bool(np.any(pbc)) else 0.0

        # a zero-volume / non-periodic cell has no strain to relax against
        target, relax_cell = atoms, _W["cell"] == "full" and volume0 > 1e-6
        if relax_cell:
            from ase.filters import FrechetCellFilter
            target = FrechetCellFilter(atoms)

        optimiser = {"FIRE": FIRE, "FIRE2": FIRE2, "LBFGS": LBFGS, "BFGS": BFGS}[_W["opt"]]
        opt = optimiser(target, logfile=None)

        traj = []
        if _W["traj"]:
            opt.attach(lambda: traj.append((atoms.get_positions().copy(),
                                            np.array(atoms.cell), float(atoms.get_potential_energy()))))
        converged = bool(opt.run(fmax=_W["fmax"], steps=_W["steps"]))

        forces = atoms.get_forces()
        result.update(
            positions=atoms.get_positions(), cell=np.array(atoms.cell),
            energy=float(atoms.get_potential_energy()), energy0=energy0, forces=forces,
            steps=int(opt.get_number_of_steps()), converged=converged,
            fmax=float(np.linalg.norm(forces, axis=1).max()),
            dmax=float(np.linalg.norm(atoms.get_positions() - positions, axis=1).max()),
            dvol=float((atoms.get_volume() - volume0) / volume0 * 100) if relax_cell else 0.0,
            cell_relaxed=relax_cell, traj=traj,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    result["seconds"] = time.perf_counter() - t0
    return result


def apply_result(atoms: Atoms, prefix: str, res: dict):
    """Write one worker result back into the structure, leaving every original label alone."""
    if res["error"]:
        atoms.info[f"{prefix}steps"], atoms.info[f"{prefix}converged"] = 0, False
        atoms.info[f"{prefix}error"] = res["error"]
    else:
        atoms.set_positions(res["positions"])
        if res["cell_relaxed"]:
            atoms.set_cell(res["cell"])
            atoms.info[f"{prefix}dvol"] = round(res["dvol"], 4)
        atoms.info[f"{prefix}energy"] = res["energy"]
        atoms.arrays[f"{prefix}forces"] = res["forces"]
        atoms.info[f"{prefix}steps"] = res["steps"]
        atoms.info[f"{prefix}converged"] = res["converged"]
        atoms.info[f"{prefix}fmax"] = round(res["fmax"], 6)
        atoms.info[f"{prefix}energy0"] = res["energy0"]
        atoms.info[f"{prefix}Erelax"] = (res["energy"] - res["energy0"]) / len(atoms)
        atoms.info[f"{prefix}dmax"] = round(res["dmax"], 4)
    atoms.info[f"{prefix}walltime"] = round(res["seconds"], 2)
    atoms.info[f"{prefix}index"] = res["index"]


def write_traj(handle, atoms: Atoms, prefix: str, res: dict):
    """All optimisation paths go into one file, tagged by structure and step."""
    for step, (positions, cell, energy) in enumerate(res.get("traj") or [], start=1):
        frame = Atoms(numbers=atoms.numbers, positions=positions, cell=cell, pbc=atoms.pbc)
        frame.info.update({f"{prefix}opt_index": res["index"], f"{prefix}opt_step": step,
                           f"{prefix}energy": energy})
        ase.io.write(handle, frame, format="extxyz")


def pick_nproc(value: str, device: str) -> int:
    """One worker per GPU by default; on CPU leave room for each worker's own threads."""
    if value != "auto":
        return max(1, int(value))
    return 1 if device != "cpu" else max(1, (os.cpu_count() or 4) // 4)

def load_part(frames: list, part: Path, prefix: str) -> set:
    """Structures already finished in a previous run are restored whole, labels included."""
    if not part.exists():
        return set()
    try:
        done_frames = ase.io.read(part, index=":")
    except Exception as exc:
        LOG(f"[warn]could not read {part} ({type(exc).__name__}) — starting over[/warn]")
        return set()
    done = set()
    for atoms in done_frames:
        index = int(atoms.info.get(f"{prefix}index", -1))
        if 0 <= index < len(frames):
            frames[index], _ = atoms, done.add(index)
    LOG(f"resuming: [hi]{len(done)}[/hi] structures already done in {part}")
    return done


def report_relax(frames: list, prefix: str, done_now: int, wall: float,
                 nproc: int, gpu_mb: float) -> dict:
    """Distributions rather than 1000 log lines."""
    def col(key, default=0.0):
        return np.array([float(a.info.get(f"{prefix}{key}", default)) for a in frames])

    steps, erelax, fmax, dmax = col("steps"), col("Erelax"), col("fmax"), col("dmax")
    ok = np.array([not a.info.get(f"{prefix}error") for a in frames])
    converged = np.array([bool(a.info.get(f"{prefix}converged", False)) for a in frames])

    rows = [("steps", "", steps), ("Erelax", "eV/atom", erelax),
            ("final fmax", "eV/Å", fmax), ("displacement", "Å", dmax)]
    LOG(table(f"relax  {prefix}", ["", "unit", "min", "median", "max", "mean"],
              [(name, unit, num(v.min(), 4), num(np.median(v), 4), num(v.max(), 4), num(v.mean(), 4))
               for name, unit, v in rows], styles=["key", "dim"]))

    rate = done_now / wall if wall else 0.0
    LOG(f"[ok]{converged.sum()}[/ok]/{len(frames)} converged · "
        f"[warn]{(~converged & ok).sum()}[/warn] hit the step limit · "
        f"[err]{(~ok).sum()}[/err] failed · {elapsed(wall)} · {rate:.2f} struct/s · "
        f"{nproc} worker(s)" + (f" · GPU peak {gpu_mb:,.0f} MB/worker" if gpu_mb else ""))

    MONITOR.report()

    stuck = [i for i, a in enumerate(frames) if not a.info.get(f"{prefix}converged", False)][:10]
    if stuck:
        LOG(f"[dim]not converged: index {', '.join(map(str, stuck))}"
            f"{' …' if (~converged).sum() > 10 else ''}[/dim]")
    for i, a in enumerate(frames):
        if a.info.get(f"{prefix}error"):
            LOG(f"[err]failed[/err] index {i}: {a.info[f'{prefix}error']}")

    return {"resources": MONITOR.summary(),
            "converged": int(converged.sum()), "failed": int((~ok).sum()), "n": len(frames),
            "steps": {"min": int(steps.min()), "median": float(np.median(steps)), "max": int(steps.max())},
            "Erelax_eV_per_atom": {"mean": float(erelax.mean()), "min": float(erelax.min())},
            "walltime_s": round(wall, 1), "struct_per_s": round(rate, 3),
            "nproc": nproc, "gpu_peak_mb_per_worker": round(gpu_mb, 1)}


def report_relax_groups(frames: list, prefix: str, groups: np.ndarray, key: str) -> dict:
    """Which family of structures is hard to relax."""
    steps = np.array([float(a.info.get(f"{prefix}steps", 0)) for a in frames])
    erelax = np.array([float(a.info.get(f"{prefix}Erelax", 0.0)) for a in frames])
    converged = np.array([bool(a.info.get(f"{prefix}converged", False)) for a in frames])

    out = {}
    for name in dict.fromkeys(groups):
        m = groups == name
        out[name] = {"N": int(m.sum()), "steps_median": float(np.median(steps[m])),
                     "converged_pct": float(100 * converged[m].mean()),
                     "Erelax_mean": float(erelax[m].mean())}
    order = sorted(out, key=lambda k: -out[k]["steps_median"])
    LOG(table(f"by {key}  (slowest first)", ["group", "N", "steps (median)", "converged", "Erelax mean"],
              [[g, f"{out[g]['N']:,}", num(out[g]["steps_median"], 1),
                f"{out[g]['converged_pct']:.0f} %", num(out[g]["Erelax_mean"], 4)] for g in order],
              styles=["key"]))
    return out


def _scatter(ax, x, y, xlabel, ylabel, groups=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    names = sorted(set(groups.tolist())) if groups is not None else []
    if 1 < len(names) <= 12:
        for i, name in enumerate(names):
            m = groups == name
            ax.scatter(x[m], y[m], color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                       alpha=0.7, s=40, edgecolors=COLORS[i % len(COLORS)], linewidths=1.5,
                       zorder=3, label=str(name))
        ax.legend(framealpha=0.5, fontsize=9, markerscale=1.2)
    else:
        ax.scatter(x, y, color=COLORS[0], marker=MARKERS[0], alpha=0.7, s=40,
                   edgecolors=COLORS[0], linewidths=1.5, zorder=3)
    ax.set(xlabel=xlabel, ylabel=ylabel)


def make_relax_plot(frames: list, prefix: str, fmax_target: float, outfile: str, groups=None):
    steps = np.array([float(a.info.get(f"{prefix}steps", 0)) for a in frames])
    erelax = np.array([float(a.info.get(f"{prefix}Erelax", 0.0)) for a in frames]) * 1000
    fmax = np.array([float(a.info.get(f"{prefix}fmax", 0.0)) for a in frames])
    natoms = np.array([len(a) for a in frames], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    _hist(axes[0, 0], steps, "Optimisation steps", COLORS[0], zero_line=False)
    _hist(axes[0, 1], erelax, "Relaxation energy (meV/atom)", COLORS[1], zero_line=True)
    _hist(axes[1, 0], np.log10(np.clip(fmax, 1e-6, None)),
          r"log$_{10}$ final max|F| (eV/$\mathrm{\AA}$)", COLORS[2], zero_line=False)
    axes[1, 0].axvline(np.log10(fmax_target), color="black", ls=LINESTYLES[1], lw=2, zorder=3)
    _scatter(axes[1, 1], natoms, steps, "Atoms per structure", "Optimisation steps", groups)

    for ax in axes.ravel():
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


def cmd_relax(args) -> int:
    import multiprocessing as mp

    t0 = time.perf_counter()
    stem = Path(args.input).stem
    outfile = args.output or str(Path(args.outdir) / f"{stem}_relax.xyz")
    device, (kind, model) = pick_device(args.device), resolve_model(args.model)
    nproc = pick_nproc(args.nproc, device)
    threads = max(1, (os.cpu_count() or 4) // nproc)

    LOG.rule(f"MFlow {__version__} · relax")
    LOG.kv({"input": args.input, "index": args.index,
            "model": model, "kind": "local file" if kind == "file" else "foundation",
            "optimiser": args.opt.upper(), "cell": args.cell,
            "fmax": f"{args.fmax} eV/Å", "max steps": args.steps,
            "workers": f"{nproc} × {threads} thread(s)", "device": device,
            "dtype": args.dtype, "head": args.head or "—",
            "output": outfile, "log": args.log})
    LOG(f"[dim]{versions()}[/dim]")

    LOG.rule("dataset")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    frames = read_dataset(args.input, args.index)
    prefix = free_prefix(frames, fix_prefix(args.prefix), args.overwrite)

    part = Path(args.outdir) / f"{stem}_relax.part.xyz"
    done = load_part(frames, part, prefix) if args.resume else set()
    if not args.resume and part.exists():
        part.unlink()
    todo = [(i, a.numbers, a.get_positions(), np.array(a.cell), a.pbc)
            for i, a in enumerate(frames) if i not in done]

    LOG.rule("relax")
    settings = {"opt": OPTIMIZERS[args.opt], "cell": args.cell, "fmax": args.fmax,
                "steps": args.steps, "traj": bool(args.traj)}
    initargs = (args.model, device, args.dtype, args.head, threads, settings)

    part_fh = open(part, "a", encoding="utf-8")
    traj_fh = open(Path(args.outdir) / args.traj, "w", encoding="utf-8") if args.traj else None
    gpu_mb, t_run = 0.0, time.perf_counter()
    try:
        with Progress(*progress_columns("structures"), console=LOG.term, transient=True) as progress:
            task = progress.add_task("relaxing", total=len(todo))

            def collect(res):
                nonlocal gpu_mb
                gpu_mb = max(gpu_mb, res.get("gpu_mb", 0.0))
                atoms = frames[res["index"]]
                apply_result(atoms, prefix, res)
                ase.io.write(part_fh, atoms, format="extxyz")
                part_fh.flush()
                if traj_fh:
                    write_traj(traj_fh, atoms, prefix, res)
                progress.advance(task)

            if nproc == 1:                       # no pool: cheaper to start and easier to debug
                _relax_init(*initargs)
                for payload in todo:
                    collect(_relax_one(payload))
            else:
                context = mp.get_context("spawn")   # fork + CUDA does not survive
                with context.Pool(nproc, initializer=_relax_init, initargs=initargs) as pool:
                    for res in pool.imap_unordered(_relax_one, todo, chunksize=1):
                        collect(res)
    finally:
        part_fh.close()
        if traj_fh:
            traj_fh.close()
            LOG(f"trajectory [key]{Path(args.outdir) / args.traj}[/key]")

    LOG.rule("results")
    ase.io.write(outfile, frames, format="extxyz")     # original order, all labels intact
    part.unlink(missing_ok=True)
    LOG(f"xyz [key]{outfile}[/key] · {len(frames)} structures · "
        f"labels {prefix}energy / {prefix}forces / [hi]{prefix}steps[/hi] / {prefix}converged")

    summary = report_relax(frames, prefix, len(todo), time.perf_counter() - t_run, nproc, gpu_mb)
    groups = group_of(frames, args.group)
    if groups is not None:
        summary["by_group"] = report_relax_groups(frames, prefix, groups, args.group)
    if not args.no_plot:
        make_relax_plot(frames, prefix, args.fmax,
                        str(Path(args.outdir) / f"{stem}_relax.png"), groups)

    peak = getrusage(RUSAGE_SELF).ru_maxrss + getrusage(RUSAGE_CHILDREN).ru_maxrss
    summary.update({"mflow": __version__, "input": os.path.abspath(args.input),
                    "output": os.path.abspath(outfile), "model": model, "prefix": prefix,
                    "optimiser": args.opt, "cell": args.cell, "fmax": args.fmax,
                    "max_steps": args.steps, "device": device, "dtype": args.dtype,
                    "peak_rss_gb": round(peak / 1e6, 2), "versions": versions()})
    json_file = Path(args.outdir) / f"{stem}_relax_metrics.json"
    json_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG(f"json [key]{json_file}[/key] · peak RSS {peak / 1e6:.1f} GB · "
        f"total [hi]{elapsed(time.perf_counter() - t0)}[/hi]")
    return 0
