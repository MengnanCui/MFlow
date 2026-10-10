# -*- coding: utf-8 -*-
"""calc / plot: batched MACE evaluation, error statistics and the summary figure."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms
from rich.progress import Progress

from . import __version__
from .data import (composition, energies, find_ref, fix_prefix, forces, free_prefix, group_of,
                   has_labels, keys, read_dataset)
from .log import LOG, MONITOR, elapsed, num, progress_columns, table
from .model import load_model, pick_device, resolve_model, versions
from .style import COLORS, HEXBIN_ABOVE, LINESTYLES, MARKERS, PLOT_MAX_POINTS, plt


def evaluate(frames: list, spec: str, batch_size: int, device: str, dtype: str,
             prefix: str, head: str | None) -> list:
    """Batched forward pass; results go into info[<prefix>energy] / arrays[<prefix>forces]."""
    import torch
    import mace.data as mdata
    from mace.tools import torch_geometric, torch_tools, utils

    model, dev = load_model(spec, device, dtype)
    ekey, fkey = keys(prefix)

    def bare(atoms):  # geometry only: keeps MACE away from the reference labels
        new = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        if head:
            new.info["head"] = head
        return new

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", None)
    dataset = [mdata.AtomicData.from_config(mdata.config_from_atoms(bare(a)), z_table=z_table,
                                            cutoff=float(model.r_max), heads=heads) for a in frames]
    loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    energy_chunks, force_chunks, t0 = [], [], time.perf_counter()
    with Progress(*progress_columns("batches"), console=LOG.term, transient=True) as progress:
        task = progress.add_task("evaluating", total=len(loader))
        for batch in loader:
            batch = batch.to(dev)
            out = model(batch.to_dict(), compute_stress=False)
            energy_chunks.append(torch_tools.to_numpy(out["energy"]))
            force_chunks += np.split(torch_tools.to_numpy(out["forces"]),
                                     batch.ptr[1:].cpu().numpy(), axis=0)[:-1]
            progress.advance(task)

    for atoms, energy, force in zip(frames, np.concatenate(energy_chunks), force_chunks):
        atoms.calc, atoms.info[ekey], atoms.arrays[fkey] = None, float(energy), np.asarray(force)

    walltime = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    LOG(f"[ok]done[/ok] {len(frames)} structures in {elapsed(walltime)} "
        f"· {len(frames) / walltime:.1f} struct/s · {sum(map(len, frames)) / walltime:.0f} atom/s "
        f"· batch {batch_size}")
    return frames


def evaluate_ase(frames: list, spec: str, device: str, dtype: str, prefix: str, head: str | None) -> list:
    """Fallback: one structure at a time through the ASE calculator."""
    kind, value = resolve_model(spec)
    kwargs = {"device": device, "default_dtype": dtype, **({"head": head} if head else {})}
    if kind == "file":
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=value, **kwargs)
    else:
        from mace.calculators import mace_mp
        calc = mace_mp(model=value, **kwargs)

    ekey, fkey = keys(prefix)
    t0 = time.perf_counter()
    with Progress(*progress_columns("structures"), console=LOG.term, transient=True) as progress:
        task = progress.add_task("evaluating", total=len(frames))
        for atoms in frames:
            work = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            work.calc = calc
            atoms.calc = None
            atoms.info[ekey] = float(work.get_potential_energy())
            atoms.arrays[fkey] = work.get_forces()
            progress.advance(task)
    LOG(f"[ok]done[/ok] {len(frames)} structures in {elapsed(time.perf_counter() - t0)} (ASE engine)")
    return frames


def run_mace(frames: list, **kwargs) -> list:
    """Batched by default; any failure (API drift, OOM) drops to the ASE calculator."""
    try:
        return evaluate(frames, **kwargs)
    except Exception as exc:
        LOG(f"[warn]batched run failed ({type(exc).__name__}: {exc}) — retrying one by one[/warn]")
        kwargs.pop("batch_size")
        return evaluate_ase(frames, **kwargs)


# ============================================================================ #
#  metrics

def stats(pred: np.ndarray, ref: np.ndarray, scale: float = 1.0) -> dict:
    """MAE / RMSE / max|Δ| (in `scale` units) and R² for one pair of arrays."""
    d = (np.asarray(pred) - np.asarray(ref)) * scale
    r2 = float(np.corrcoef(ref, pred)[0, 1] ** 2) if len(ref) > 1 and np.ptp(ref) else float("nan")
    return {"MAE": float(np.abs(d).mean()), "RMSE": float(np.sqrt((d ** 2).mean())),
            "max": float(np.abs(d).max()), "R2": r2, "N": int(d.size)}


def align_e0(frames: list, diff: np.ndarray, mode) -> tuple[np.ndarray, np.ndarray, list]:
    """
    No two codes share an atomic reference, so total energies are only comparable up to
    one constant per element:  E_pred - E_ref = Σ_j n_j δ_j.  This returns δ (eV/atom of
    that element), fitted from the dataset itself — no isolated-atom energies needed.

        fit  : least squares over the composition matrix (default; needs independent compositions)
        mean : a single δ shared by all elements — what a plain mean shift does
        none : δ = 0, compare the raw numbers
        dict : δ supplied by the user, {"Si": -0.12, ...} or {"14": -0.12, ...}
    """
    from ase.data import chemical_symbols
    C, zs = composition(frames)
    if mode == "none":
        return np.zeros(len(zs)), C, zs
    if isinstance(mode, dict):
        delta = np.array([float(mode.get(chemical_symbols[z], mode.get(str(z), 0.0))) for z in zs])
    elif mode == "fit":
        delta, _, rank, _ = np.linalg.lstsq(C, diff, rcond=None)  # minimum-norm solution
        if rank < len(zs):
            LOG(f"[warn]E0 fit: only {rank} of {len(zs)} element shifts are identifiable — the total "
                f"correction is still exact, but the per-element numbers below are one of many solutions[/warn]")
        if len(frames) < 3 * len(zs):
            LOG(f"[warn]E0 fit on {len(frames)} structures for {len(zs)} elements — the fit can absorb "
                f"real error; check against '-e0 none'[/warn]")
    else:
        delta = np.full(len(zs), float((diff / C.sum(1)).mean()))
    return delta, C, zs


def compare(frames: list, pred: str, ref: str, e0="fit") -> tuple[dict, dict]:
    """Errors (meV/atom, meV/Å) after E0 alignment + the arrays the figure needs."""
    from ase.data import chemical_symbols
    n = np.array([len(a) for a in frames], dtype=float)
    et_pred, et_ref = energies(frames, pred), energies(frames, ref)  # total energies, eV
    delta, C, zs = align_e0(frames, et_pred - et_ref, e0)

    series = {"e_pred": (et_pred - C @ delta) / n, "e_ref": et_ref / n, "n_atoms": n}
    series["dE"] = (series["e_pred"] - series["e_ref"]) * 1000            # meV/atom, signed
    errors = {"energy": stats(series["e_pred"], series["e_ref"], 1000),
              "energy_raw": stats(et_pred / n, et_ref / n, 1000),
              "e0_mode": e0 if isinstance(e0, str) else "user",
              "e0_shift_eV": {chemical_symbols[z]: float(d) for z, d in zip(zs, delta)}}
    if keys(ref)[1] in frames[0].arrays:
        series["f_pred"], series["f_ref"] = forces(frames, pred), forces(frames, ref)
        errors["forces"] = stats(series["f_pred"], series["f_ref"], 1000)  # forces need no alignment
        fp, fr = keys(pred)[1], keys(ref)[1]
        sq = np.array([np.sum((a.arrays[fp] - a.arrays[fr]) ** 2) for a in frames])  # eV²/Å²
        series["dF"] = np.sqrt(sq / (3 * n)) * 1000                        # meV/Å, per structure
        series["sqF"] = sq
    return errors, series


def report_errors(errors: dict, pred: str, ref: str):
    rows = [(name, num(s["MAE"]), num(s["RMSE"]), num(s["max"]), f"{s['R2']:.4f}", f"{s['N']:,}", unit)
            for name, s, unit in [("energy (E0 aligned)", errors["energy"], "meV/atom"),
                                  ("energy (raw)", errors["energy_raw"], "meV/atom")]
            + ([("forces", errors["forces"], "meV/Å")] if "forces" in errors else [])]
    LOG(table(f"error  {pred or 'pred'} vs {ref or 'ref'}",
              ["", "MAE", "RMSE", "max|Δ|", "R²", "N", "unit"], rows,
              styles=["key", None, "hi", None, None, "dim", "dim"]))
    shifts = " · ".join(f"{el} {d:+.4f}" for el, d in errors["e0_shift_eV"].items())
    LOG(f"[dim]E0 alignment ({errors['e0_mode']}), eV per atom of element: {shifts}[/dim]")


def report_prediction(frames: list, prefix: str) -> dict:
    n = np.array([len(a) for a in frames], dtype=float)
    e_pa = energies(frames, prefix) / n
    fmax = np.array([np.linalg.norm(a.arrays[keys(prefix)[1]], axis=1).max() for a in frames])
    rows = [("energy", "eV/atom", e_pa), ("max |F|", "eV/Å", fmax)]
    LOG(table(f"prediction  {prefix or ''}", ["", "unit", "min", "max", "mean", "std"],
              [(name, unit, num(v.min(), 4), num(v.max(), 4), num(v.mean(), 4), num(v.std(), 4))
               for name, unit, v in rows], styles=["key", "dim"]))
    return {name: {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                   "std": float(v.std()), "unit": unit} for name, unit, v in rows}

def report_groups(groups: np.ndarray, series: dict, key: str) -> dict:
    """One row per group, worst energy RMSE first — the fast way to see which family fails."""
    out = {}
    for name in dict.fromkeys(groups):
        m = groups == name
        row = {"N": int(m.sum()),
               "E_MAE": float(np.abs(series["dE"][m]).mean()),
               "E_RMSE": float(np.sqrt(np.mean(series["dE"][m] ** 2)))}
        if "sqF" in series:
            row["F_RMSE"] = float(np.sqrt(series["sqF"][m].sum() / (3 * series["n_atoms"][m]).sum()) * 1000)
        out[name] = row

    order = sorted(out, key=lambda k: -out[k]["E_RMSE"])
    has_f = "sqF" in series
    LOG(table(f"by {key}  (worst first)",
              ["group", "N", "E MAE", "E RMSE", "F RMSE"] if has_f else ["group", "N", "E MAE", "E RMSE"],
              [[g, f"{out[g]['N']:,}", num(out[g]["E_MAE"]), num(out[g]["E_RMSE"])]
               + ([num(out[g]["F_RMSE"])] if has_f else []) for g in order],
              styles=["key"]))
    LOG("[dim]E in meV/atom, F in meV/Å · the E0 shift is fitted once on the whole dataset, "
        "so the groups stay comparable[/dim]")
    return out


def report_worst(frames: list, series: dict, groups: np.ndarray | None, top: int = 5):
    """The structures to look at first, by |ΔE| per atom."""
    order = np.argsort(-np.abs(series["dE"]))[:top]
    has_f = "dF" in series
    LOG(table(f"worst {len(order)} structures by |ΔE|",
              ["#", "group", "formula", "atoms", "ΔE", "F RMSE"] if has_f else
              ["#", "group", "formula", "atoms", "ΔE"],
              [[i, groups[i] if groups is not None else "—",
                frames[i].get_chemical_formula(mode="hill", empirical=True), len(frames[i]),
                f"{series['dE'][i]:+.2f}"] + ([num(series["dF"][i])] if has_f else [])
               for i in order], styles=["dim", "key"]))
    LOG("[dim]index is the position in the input file · ΔE in meV/atom, F RMSE in meV/Å[/dim]")

def _thin(*arrays):
    if arrays[0].size <= PLOT_MAX_POINTS:
        return arrays
    idx = np.random.default_rng(8465).choice(arrays[0].size, PLOT_MAX_POINTS, replace=False)
    LOG(f"[dim]plot: random subsample {PLOT_MAX_POINTS:,} / {arrays[0].size:,} points[/dim]")
    return [a[idx] for a in arrays]


def _parity(ax, x, y, xlabel, ylabel, title, note, groups=None):
    x, y, groups = _thin(np.asarray(x, float), np.asarray(y, float),
                         np.asarray(groups) if groups is not None else np.zeros(len(x)))
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.05 or 1.0
    names = sorted(set(groups.tolist()))
    size = 80 if x.size <= 500 else (40 if x.size <= 3000 else 12)  # readable when points overlap
    if 1 < len(names) <= 12:
        for i, name in enumerate(names):
            m = groups == name
            ax.scatter(x[m], y[m], color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                       alpha=0.7, s=size, edgecolors=COLORS[i % len(COLORS)], label=str(name),
                       linewidths=1.5 if size >= 40 else 0.5, zorder=3)
        ax.legend(framealpha=0.5, fontsize=9, loc="lower right", markerscale=1.2)
    elif x.size > HEXBIN_ABOVE:
        from matplotlib.colors import LogNorm
        hb = ax.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1, norm=LogNorm())
        ax.figure.colorbar(hb, ax=ax, label="counts")
    else:
        ax.scatter(x, y, color=COLORS[0], marker=MARKERS[0], alpha=0.7, s=size,
                   edgecolors=COLORS[0], linewidths=1.5 if size >= 40 else 0.5, zorder=3)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k", ls=LINESTYLES[1], lw=2, zorder=4)
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel=xlabel, ylabel=ylabel, title=title)
    ax.text(0.04, 0.96, note, transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6, ec="none"))


def _bars(ax, names: list, values: list, xlabel: str, color: str):
    """Horizontal so the group names stay readable; worst group on top."""
    order = np.argsort(values)
    ax.barh([str(names[i]) for i in order], [values[i] for i in order], color=color,
            alpha=0.8, edgecolor="black", linewidth=1.0, zorder=2)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=11)


def _hist(ax, values, xlabel, color, zero_line=True):
    ax.hist(_thin(np.asarray(values))[0], bins=50, color=color, alpha=0.75,
            edgecolor="black", linewidth=1.0, zorder=2)
    if zero_line:
        ax.axvline(0, color="black", ls=LINESTYLES[1], lw=2, zorder=3)
    ax.set(xlabel=xlabel, ylabel="Counts")


def make_plot(frames: list, pred: str, ref: str | None, errors: dict | None,
              series: dict | None, outfile: str, groups=None, by_group: dict | None = None):
    """
    No reference        : 1x2 prediction distributions.
    Reference           : 2x2 parity + error histograms.
    Reference + -group  : 2x3, the extra column ranks the groups by RMSE.
    """
    tag = lambda p: (p or "pred").rstrip("_").upper()

    if ref is None or errors is None:
        n = np.array([len(a) for a in frames], dtype=float)
        fmax = np.array([np.linalg.norm(a.arrays[keys(pred)[1]], axis=1).max() for a in frames])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _hist(axes[0], energies(frames, pred) / n, "Energy (eV/atom)", COLORS[0], zero_line=False)
        _hist(axes[1], fmax, r"max |F| (eV/$\mathrm{\AA}$)", COLORS[1], zero_line=False)
    else:
        # energies here are already E0-aligned, so the axes are directly comparable
        e_pred, e_ref, es = series["e_pred"], series["e_ref"], errors["energy"]
        ncols = 3 if by_group else 2
        fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols + 0.5, 8.5))
        _parity(axes[0, 0], e_ref, e_pred, f"{tag(ref)} energy (eV/atom)",
                f"{tag(pred)} energy (eV/atom)", f"Energy · E0 {errors['e0_mode']}",
                f"RMSE = {es['RMSE']:.2f} meV/atom\nR$^2$ = {es['R2']:.4f}\nN = {es['N']:,}", groups)
        _hist(axes[1, 0], series["dE"], "Energy error (meV/atom)", COLORS[0])
        if "forces" in errors:
            f_pred, f_ref, fs = series["f_pred"], series["f_ref"], errors["forces"]
            # one group label per force component
            per_comp = None if groups is None else np.repeat(groups, (3 * series["n_atoms"]).astype(int))
            _parity(axes[0, 1], f_ref, f_pred, rf"{tag(ref)} force comp. (eV/$\mathrm{{\AA}}$)",
                    rf"{tag(pred)} force comp. (eV/$\mathrm{{\AA}}$)", "Forces",
                    f"RMSE = {fs['RMSE']:.1f} meV/$\\mathrm{{\\AA}}$\nR$^2$ = {fs['R2']:.4f}\nN = {fs['N']:,}",
                    per_comp)
            _hist(axes[1, 1], (f_pred - f_ref) * 1000, r"Force error (meV/$\mathrm{\AA}$)", COLORS[1])
        else:
            axes[0, 1].axis("off"), axes[1, 1].axis("off")

        if by_group:
            names = list(by_group)
            _bars(axes[0, 2], names, [by_group[g]["E_RMSE"] for g in names],
                  "Energy RMSE (meV/atom)", COLORS[0])
            if "F_RMSE" in by_group[names[0]]:
                _bars(axes[1, 2], names, [by_group[g]["F_RMSE"] for g in names],
                      r"Force RMSE (meV/$\mathrm{\AA}$)", COLORS[1])
            else:
                axes[1, 2].axis("off")

    for ax in np.atleast_1d(axes).ravel():
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")

def tag_structures(frames: list, prefix: str, series: dict):
    """
    Per-structure diagnostics into info, so the file itself can be sorted, filtered or
    fed back to `-group <prefix>dE`:  dE in meV/atom (signed), dF in meV/Å (RMSE).
    """
    for i, atoms in enumerate(frames):
        atoms.info[f"{prefix}dE"] = float(series["dE"][i])
        if "dF" in series:
            atoms.info[f"{prefix}dF"] = float(series["dF"][i])
    extra = f" / {prefix}dF" if "dF" in series else ""
    LOG(f"[dim]per-structure diagnostics written as {prefix}dE{extra}[/dim]")


def parse_e0(value: str):
    """'fit' | 'mean' | 'none' as is; anything else is a json file of per-element shifts."""
    return value if value in ("fit", "mean", "none") else json.loads(Path(value).read_text(encoding="utf-8"))


def resolve_ref(frames: list, requested: str | None, pred: str) -> str | None:
    if requested is None:
        ref = find_ref(frames, exclude=pred)
        LOG(f"reference [key]{keys(ref)[0]}[/key] (auto)" if ref is not None else
            "[warn]no reference labels found — prediction only[/warn]")
        return ref
    ref = fix_prefix(requested)
    if has_labels(frames[0], ref, forces=False):
        return ref
    LOG(f"[warn]{keys(ref)[0]} not in the dataset — prediction only[/warn]")
    return None


def cmd_calc(args) -> int:
    t0 = time.perf_counter()
    prefix, stem = fix_prefix(args.prefix), Path(args.input).stem
    outfile = args.output or str(Path(args.outdir) / f"{stem}_mace.xyz")
    device, (kind, model) = pick_device(args.device), resolve_model(args.model)

    LOG.rule(f"MFlow {__version__} · calc")
    LOG.kv({"input": args.input, "index": args.index,
            "model": model, "kind": "local file" if kind == "file" else "foundation",
            "batch size": args.batch_size, "device": device,
            "dtype": args.dtype, "head": args.head or "—",
            "labels": " / ".join(keys(prefix)), "reference": args.ref_prefix or "auto",
            "E0 align": args.e0, "group by": args.group or "—",
            "output": outfile, "log": args.log})
    LOG(f"[dim]{versions()}[/dim]")

    LOG.rule("dataset")
    frames = read_dataset(args.input, args.index)
    prefix = free_prefix(frames, prefix, args.overwrite)
    ref = resolve_ref(frames, args.ref_prefix, prefix)

    LOG.rule("mace")
    frames = run_mace(frames, spec=args.model, batch_size=args.batch_size,
                      device=device, dtype=args.dtype, prefix=prefix, head=args.head)

    LOG.rule("results")
    MONITOR.report()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    prediction = report_prediction(frames, prefix)

    errors, series = compare(frames, prefix, ref, parse_e0(args.e0)) if ref is not None else (None, None)
    groups = by_group = None
    if errors:
        report_errors(errors, prefix, ref)
        tag_structures(frames, prefix, series)     # per-structure ΔE / ΔF into the file
        groups = group_of(frames, args.group)
        if groups is not None:
            by_group = report_groups(groups, series, args.group)
        report_worst(frames, series, groups)

    ase.io.write(outfile, frames, format="extxyz")
    LOG(f"xyz [key]{outfile}[/key] · {len(frames)} structures")
    if not args.no_plot:
        make_plot(frames, prefix, ref, errors, series,
                  str(Path(args.outdir) / f"{stem}_summary.png"), groups, by_group)

    metrics = {"mflow": __version__, "time": datetime.now().isoformat(timespec="seconds"),
               "input": os.path.abspath(args.input), "output": os.path.abspath(outfile),
               "model": model, "batch_size": args.batch_size, "device": device, "dtype": args.dtype,
               "prefix": prefix, "ref_prefix": ref, "prediction": prediction, "errors": errors,
               "group_by": args.group, "by_group": by_group, "resources": MONITOR.summary(),
               "versions": versions()}
    json_file = Path(args.outdir) / f"{stem}_metrics.json"
    json_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG(f"json [key]{json_file}[/key] · total [hi]{elapsed(time.perf_counter() - t0)}[/hi]")
    return 0


def cmd_plot(args) -> int:
    pred, stem = fix_prefix(args.pred_prefix), Path(args.input).stem
    LOG.rule(f"MFlow {__version__} · plot")
    frames = read_dataset(args.input, args.index)
    if not has_labels(frames[0], pred, forces=False):
        LOG(f"[err]{keys(pred)[0]} not found in {args.input}[/err]")
        return 1

    ref = resolve_ref(frames, args.ref_prefix, pred)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    errors, series = compare(frames, pred, ref, parse_e0(args.e0)) if ref is not None else (None, None)
    groups = by_group = None
    if errors:
        report_errors(errors, pred, ref)
        groups = group_of(frames, args.group)
        if groups is not None:
            by_group = report_groups(groups, series, args.group)
        report_worst(frames, series, groups)
        (Path(args.outdir) / f"{stem}_metrics.json").write_text(
            json.dumps({"errors": errors, "group_by": args.group, "by_group": by_group}, indent=2),
            encoding="utf-8")
    make_plot(frames, pred, ref, errors, series,
              str(Path(args.outdir) / f"{stem}_summary.png"), groups, by_group)
    return 0
