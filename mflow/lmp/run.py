# -*- coding: utf-8 -*-
"""
mflow md — MACE molecular dynamics where LAMMPS does the dynamics.

    1. inputs      data.lmp + in.lammps from one template, element order taken from deploy.json
    2. skiprun     `lmp -sr`: the whole input is parsed and set up, nothing is integrated
    3. check       `run 0` on the start structure; LAMMPS energy / forces / pressure must match
                   the same model evaluated outside LAMMPS (element mapping, units, virial sign)
    4. run         lmp as a subprocess, log.lammps followed live
    5. report      the md-report gates

The run directory holds everything needed to reproduce or resume the run.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from rich.progress import Progress

from .. import __version__
from ..data import read_dataset
from ..log import LOG, MONITOR, elapsed, progress_columns, table
from ..model import pick_device, versions
from ..report import Report
from ..thresholds import section
from . import env
from .deploy import load_deploy
from .inputs import MDParams, check_input, md_input, read_back, sanity, species_for, write_data

# consistency tolerances by export precision: energy eV/atom, force eV/Å, pressure (abs bar, relative)
TOL = {k: tuple(v) for k, v in section("consistency").items()}   # mflow/thresholds.yaml


# ============================================================================ #
#  consistency check
# ============================================================================ #

def reference_calculator(ref: dict, device: str):
    """ASE calculator that should reproduce the LAMMPS pair style (deploy.json 'reference')."""
    kind = ref.get("type")
    if kind == "mace":
        from ..model import build_calculator
        return build_calculator(ref["model"], device, ref.get("dtype", "float64"), ref.get("head"))
    if kind == "lj":                      # used by the test-suite with pair_style lj/cut + shift
        from ase.calculators.lj import LennardJones
        return LennardJones(epsilon=ref["epsilon"], sigma=ref["sigma"], rc=ref["rc"], smooth=False)
    raise ValueError(f"unknown reference type {kind!r}")


def read_check_dump(path: Path) -> np.ndarray:
    """Forces sorted by atom id from `write_dump ... id type x y z fx fy fz`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    cols = lines[start].split()[2:]
    data = np.array([[float(x) for x in l.split()] for l in lines[start + 1:] if l.strip()])
    data = data[np.argsort(data[:, cols.index("id")])]
    return data[:, [cols.index("fx"), cols.index("fy"), cols.index("fz")]]


def consistency(run_dir: Path, deploy: dict, species: list, natoms: int, rep: Report) -> dict:
    from ase import units
    from ..analysis.thermo import parse_log

    log = parse_log(run_dir / "check.log")
    if not log["runs"]:
        rep.check("consistency", "fail", "run 0 produced no thermo output",
                  value=log["errors"][:3], suggest="see check.log / check.out")
        return {}
    cols, row = log["runs"][0]["columns"], log["runs"][0]["data"][0]
    e_lmp, p_lmp = row[cols.index("PotEng")], row[cols.index("Press")]
    f_lmp = read_check_dump(run_dir / "check_forces.dump")

    atoms = read_back(run_dir / "data.lmp", species)
    ref = deploy.get("reference") or {}
    atoms.calc = reference_calculator(ref, pick_device("auto"))
    e_ref = float(atoms.get_potential_energy())
    f_ref = atoms.get_forces()
    p_ref = -float(np.trace(atoms.get_stress(voigt=False))) / 3 / units.bar

    tol_e, tol_f, tol_p_abs, tol_p_rel = TOL.get(ref.get("dtype", deploy.get("dtype", "float64")), TOL["float32"])
    de = abs(e_lmp - e_ref) / natoms
    df = float(np.abs(f_lmp - f_ref).max())
    dp = abs(p_lmp - p_ref)
    tol_p = max(tol_p_abs, tol_p_rel * abs(p_ref))
    LOG(table("LAMMPS vs reference (run 0)", ["", "LAMMPS", "reference", "|Δ|", "tolerance", "unit"],
              [("energy / atom", f"{e_lmp / natoms:.8f}", f"{e_ref / natoms:.8f}", f"{de:.2e}", f"{tol_e:g}", "eV"),
               ("max force comp.", f"{np.abs(f_lmp).max():.6f}", f"{np.abs(f_ref).max():.6f}", f"{df:.2e}",
                f"{tol_f:g}", "eV/Å"),
               ("pressure", f"{p_lmp:.3f}", f"{p_ref:.3f}", f"{dp:.3g}", f"{tol_p:.3g}", "bar")],
              styles=["key", None, None, "hi", "dim", "dim"]))
    ok_e, ok_f, ok_p = de <= tol_e, df <= tol_f, dp <= tol_p
    hint = ("element order / type mapping or the exported head is wrong" if not ok_e or not ok_f else
            "pressure / virial convention or units differ")
    rep.check("consistency: energy", "pass" if ok_e else "fail", f"|ΔE| {de:.2e} eV/atom", value=de,
              suggest=f"LAMMPS and the reference model disagree: {hint}")
    rep.check("consistency: forces", "pass" if ok_f else "fail", f"max |ΔF| {df:.2e} eV/Å", value=df,
              suggest=f"LAMMPS and the reference model disagree: {hint}")
    rep.check("consistency: pressure", "pass" if ok_p else "fail", f"|ΔP| {dp:.3g} bar", value=dp,
              suggest=f"LAMMPS and the reference model disagree: {hint}")
    return {"dE_eV_atom": de, "dF_max_eV_A": df, "dP_bar": dp, "E_lmp": e_lmp, "E_ref": e_ref,
            "P_lmp_bar": p_lmp, "P_ref_bar": p_ref}


# ============================================================================ #
#  live run
# ============================================================================ #

def follow(proc, log_file: Path, total_steps: int, start_step: int = 0, dt_fs: float = 1.0) -> None:
    """Progress from the thermo lines LAMMPS appends to log.lammps."""
    pos, cols = log_file.stat().st_size if log_file.exists() else 0, None
    t0, s0 = time.perf_counter(), None
    with Progress(*progress_columns("steps"), console=LOG.term, transient=True) as progress:
        task = progress.add_task("lammps", total=total_steps, completed=start_step)
        while True:
            done = proc.poll() is not None
            if log_file.exists():
                with open(log_file, encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                cut = chunk.rfind("\n")
                if cut >= 0:
                    pos += len(chunk[:cut + 1].encode("utf-8"))
                    for line in chunk[:cut].splitlines():
                        parts = line.split()
                        if parts and parts[0] == "Step":
                            cols = parts
                        elif cols and len(parts) == len(cols) and parts[0].isdigit():
                            step = int(parts[0])
                            s0 = step if s0 is None else s0
                            wall = time.perf_counter() - t0
                            rate = ""
                            if wall > 0 and step > s0:          # fs/s -> ns/day
                                rate = f" · {(step - s0) * dt_fs / wall * 86400 * 1e-6:.2f} ns/day"
                            temp = f" · T {float(parts[cols.index('Temp')]):.0f} K" if "Temp" in cols else ""
                            progress.update(task, completed=step, description=f"lammps{temp}{rate}")
            if done:
                return
            time.sleep(1.0)


def skiprun_input(p: MDParams) -> str:
    """The production input with its own log/dump and no restart files — -sr must leave no trace."""
    text = md_input(p, log_file="skiprun.log").replace("traj.lammpstrj", "skiprun.lammpstrj")
    return "\n".join(l for l in text.splitlines() if not l.startswith(("restart ", "write_restart"))) + "\n"


def newest_restart(run_dir: Path) -> Path | None:
    cands = [p for p in (run_dir / "restart.a", run_dir / "restart.b", run_dir / "final.restart") if p.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


# ============================================================================ #
#  command
# ============================================================================ #

def cmd_md(args) -> int:
    t0 = time.perf_counter()
    run_dir = Path(args.outdir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    deploy = load_deploy(args.deploy)
    rep = Report("md", run_dir=str(run_dir), deploy=deploy["_path"])

    LOG.rule(f"MFlow {__version__} · md")
    frames = read_dataset(args.input, args.index)
    atoms = frames[-1]
    if len(frames) > 1:
        LOG(f"[dim]{len(frames)} structures in the selection — using the last one[/dim]")
    try:
        species = species_for([atoms], deploy["elements"])
    except ValueError as exc:
        rep.check("elements", "fail", str(exc))
        return _finish(rep, run_dir)
    if not any(atoms.pbc) and abs(atoms.cell.volume) < 1e-6:
        rep.check("cell", "fail", "non-periodic structure without a box",
                  suggest="give the structure a cell (atoms.center(vacuum=10)) before running LAMMPS")
        return _finish(rep, run_dir)

    p = MDParams(ensemble=args.ensemble, temp=args.temp, temp_end=args.temp_end, press=args.press,
                 dt=args.dt, steps=args.steps, tdamp=args.tdamp, pdamp=args.pdamp, seed=args.seed,
                 dump=args.dump, thermo=args.thermo, restart=args.restart, species=species,
                 pair_style=deploy["pair_style"], pair_coeff=deploy.get("pair_coeff", "* * {species}"),
                 pair_extra=deploy.get("pair_extra", []), periodic=tuple(bool(x) for x in atoms.pbc))
    kokkos = bool(deploy.get("kokkos", True))
    LOG.kv({"structure": f"{args.input} [{args.index}]", "atoms": len(atoms), "species (types)": " ".join(species),
            "ensemble": p.ensemble, "T": f"{p.temp:g}→{p.temp_end:g} K", "P": f"{p.press:g} bar" if p.ensemble == "npt" else "—",
            "dt": f"{p.dt:g} fs", "steps": f"{p.steps:,} ({p.steps * p.dt / 1000:g} ps)",
            "model": Path(deploy.get("exported", deploy["pair_style"])).name, "head": deploy.get("head") or "—",
            "precision": deploy.get("dtype", "?"), "GPUs": args.ngpu, "run dir": str(run_dir)})
    LOG(f"[dim]{versions()}[/dim]")

    LOG.rule("inputs")
    sanity(p, atoms, deploy.get("r_max"), args.ngpu if kokkos else 0, rep)
    resume = newest_restart(run_dir) if args.resume else None
    if args.resume and not resume:
        rep.check("resume", "fail", f"no restart file in {run_dir}")
        return _finish(rep, run_dir)
    if not resume:
        write_data(atoms, run_dir / "data.lmp", species)
        (run_dir / "log.lammps").unlink(missing_ok=True)
    infile = "in.resume" if resume else "in.lammps"
    (run_dir / infile).write_text(md_input(p, resume_from=resume.name if resume else None), encoding="utf-8")
    (run_dir / "in.check").write_text(check_input(p), encoding="utf-8")
    (run_dir / "in.skiprun").write_text(skiprun_input(p), encoding="utf-8")
    meta = {"mflow": __version__, "params": p.to_dict(), "natoms": len(atoms), "input": str(Path(args.input).resolve()),
            "index": args.index, "deploy": deploy["_path"], "kokkos": kokkos, "ngpu": args.ngpu}
    (run_dir / "md.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG(f"inputs in [key]{run_dir}[/key]: data.lmp · {infile} · in.check")
    if rep.failed and not args.force:
        return _finish(rep, run_dir)

    exe = env.find_lmp(args.lmp)
    if not exe:
        rep.check("lmp", "fail", "no LAMMPS executable found",
                  suggest="pass -lmp /path/to/lmp or export MFLOW_LMP=/path/to/lmp")
        return _finish(rep, run_dir)
    info = env.lmp_info(exe)
    rep.data["lammps"] = {k: info[k] for k in ("exe", "version", "kokkos", "kokkos_precision")}
    missing = env.missing_for_mliap(info) if kokkos else []
    rep.check("lammps build", "fail" if missing else "pass",
              f"missing: {', '.join(missing)}" if missing else f"{info['version']} · {info['kokkos'] or 'no KOKKOS'}",
              suggest="this LAMMPS cannot run MACE through ML-IAP: rebuild with KOKKOS (CUDA), ML-IAP, "
                      "ML-SNAP, PYTHON and MLIAP_ENABLE_PYTHON (see `mflow lmp-check`)")
    if args.dry or (rep.failed and not args.force):
        if args.dry:
            LOG("[ok]dry run[/ok] — inputs written, LAMMPS not started")
        return _finish(rep, run_dir)

    # -- skiprun: parse + setup of the real input ------------------------------------ #
    LOG.rule("skiprun (lmp -sr)")
    cmd = env.command(exe, "in.skiprun", args.ngpu, kokkos, skiprun=True)
    LOG(f"[dim]{' '.join(cmd)}[/dim]")
    proc = env.run(cmd, run_dir, run_dir / "skiprun.out", timeout=1800)
    errs = env.lammps_errors(proc.stdout + proc.stderr)
    rep.check("skiprun", "fail" if proc.returncode or errs else "pass",
              errs[0] if errs else (f"exit {proc.returncode}" if proc.returncode else "input parsed and set up"),
              value=errs[:5], suggest="the input does not run: see skiprun.out")
    if rep.failed:
        return _finish(rep, run_dir)

    # -- consistency check ------------------------------------------------------------ #
    if not args.nocheck and not resume:
        LOG.rule("consistency check (run 0)")
        cmd = env.command(exe, "in.check", 1, kokkos)
        proc = env.run(cmd, run_dir, run_dir / "check.out", timeout=1800)
        errs = env.lammps_errors(proc.stdout + proc.stderr)
        if proc.returncode or errs:
            rep.check("consistency", "fail", errs[0] if errs else f"exit {proc.returncode}",
                      suggest="run 0 failed: see check.out")
        else:
            rep.data["consistency"] = consistency(run_dir, deploy, species, len(atoms), rep)
        if rep.failed and not args.force:
            return _finish(rep, run_dir)

    # -- production -------------------------------------------------------------------- #
    LOG.rule("lammps")
    cmd = env.command(exe, infile, args.ngpu, kokkos)
    LOG(f"[dim]{' '.join(cmd)}[/dim]")
    start = 0
    if resume:
        from ..analysis.thermo import merged, parse_log
        th = merged(parse_log(run_dir / "log.lammps")) if (run_dir / "log.lammps").exists() else {}
        start = int(th["Step"][-1]) if th else 0
    t_run = time.perf_counter()
    with open(run_dir / "lammps.out", "a", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            follow(proc, run_dir / "log.lammps", p.steps, start, p.dt)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            LOG("[warn]interrupted — continue later with -resume[/warn]")
            raise
    code = proc.wait()
    MONITOR.report()
    rep.data["walltime_s"] = round(time.perf_counter() - t_run, 1)
    rep.data["resources"] = MONITOR.summary()
    rep.check("lammps exit", "pass" if code == 0 else "fail", f"exit {code} after {elapsed(rep.data['walltime_s'])}",
              suggest="LAMMPS exited with an error: see lammps.out and log.lammps")

    LOG.rule("analysis")
    from ..analysis.report import analyse_md
    rep.data.update(analyse_md(run_dir, rep, skip=args.skip, stride=1, plot=not args.no_plot))
    LOG(f"total [hi]{elapsed(time.perf_counter() - t0)}[/hi]")
    return _finish(rep, run_dir)


def _finish(rep: Report, run_dir: Path) -> int:
    LOG.rule("verdict")
    rep.show()
    rep.write(run_dir / "md_report.json")
    return 0 if not rep.failed else 2
