# -*- coding: utf-8 -*-
"""
Explorers — propose structures. Each returns a list of Atoms tagged with
info["mflow_source"] (which explorer) and whatever conditions produced it (T, P, ...).

    md       short MD on a T (and P) grid from every start structure; the grid widens by
             `widen` each round so exploration moves outwards as the model gets better.
             engine "lammps": through `mflow md` (needs a deploy.json: path, or "auto" to export
             committee member 0); engine "ase": Langevin NVT with the MACE ASE calculator.
    perturb  rattle + random strain (+ vacancies) around every start structure.
    minhop   ASE minima hopping with the MACE calculator; minima and the MD between them.
  v0.3 (all off by default, switch on in campaign.yaml):
    regions     explore.md.regions — DP-GEN-style stages explored in order; a region is left when it
                passes the L1 criteria (mflow/al/convergence.py); MD length grows by `grow` per round
    drivers     explore.md.drivers — N committee members each drive their own MD (DP-GEN style)
    heatquench  heat to T_high, then a linear quench to T_low (Langevin)
    rss         random structure search (AIRSS-lite): random cells at the system's composition,
                relaxed with the model (positions + cell); relaxation paths are kept
    ubmd        uncertainty-biased MD on the committee (mflow/al/bias.py) — pushes towards where the
                members disagree; may reach unphysical regions, the failed threshold catches those
    plumed      LAMMPS MD with a PLUMED bias (fix plumed; needs the PLUMED package in lmp)
"""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms

from ..log import LOG
from ..model import build_calculator, pick_device


def tag(frames: list, **info) -> list:
    for a in frames:
        a.calc = None
        a.info.update(info)
    return frames


def bare(atoms: Atoms) -> Atoms:
    return Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)


# ============================================================================ #
#  perturbations
# ============================================================================ #

def explore_perturb(system: list, cfg: dict, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for si, start in enumerate(system):
        for k in range(int(cfg["n"])):
            a = bare(start)
            if cfg.get("strain") and any(a.pbc):
                eps = np.eye(3) + np.diag(rng.uniform(-cfg["strain"], cfg["strain"], 3))
                a.set_cell(np.array(a.cell) @ eps, scale_atoms=True)
            if cfg.get("vacancies") and len(a) > cfg["vacancies"] + 1:
                for i in sorted(rng.choice(len(a), size=int(cfg["vacancies"]), replace=False), reverse=True):
                    del a[int(i)]
            a.rattle(stdev=float(cfg.get("rattle", 0.05)), rng=rng)
            out.append(a)
    LOG(f"perturb: {len(out)} structures ({cfg['n']} per start structure)")
    return tag(out, mflow_source="perturb")


# ============================================================================ #
#  MD
# ============================================================================ #

def md_grid(cfg: dict, round_: int) -> list[tuple[float, float]]:
    widen = float(cfg.get("widen", 1.0)) ** max(round_ - 1, 0)
    temps = [float(t) * widen for t in cfg["temps"]]
    return [(t, float(p)) for t in temps for p in cfg.get("press", [1.0])]


def region_cfg(md: dict, region: dict, k: int) -> tuple[dict, list[tuple[float, float]]]:
    """MD settings of one region in its k-th round: its temps/press, steps × grow^k (≤ max_steps)."""
    cfg = {**md, **{key: v for key, v in region.items() if key not in ("name", "grow", "max_steps")}}
    steps = float(cfg["steps"]) * float(region.get("grow", 1.0)) ** k
    if region.get("max_steps"):
        steps = min(steps, float(region["max_steps"]))
    cfg["steps"] = int(steps)
    return cfg, [(float(t), float(p)) for t in cfg["temps"] for p in cfg.get("press", [1.0])]


def md_blew_up(a: Atoms, min_dist: float = 0.5) -> bool:
    """Atoms closer than min_dist Å, or non-finite positions — the trajectory is no longer physical."""
    if not np.all(np.isfinite(a.positions)):
        return True
    if len(a) < 2 or len(a) > 2000:
        return False
    d = a.get_all_distances(mic=any(a.pbc))
    return bool(d[np.triu_indices(len(a), 1)].min() < min_dist)


def explore_md_ase(system: list, cfg: dict, model: str, round_: int, workdir: Path, seed: int,
                   grid: list | None = None, calc=None, stats: dict | None = None, label: str = "md") -> list:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    calc = calc or build_calculator(model, pick_device("auto"), "float64", None)
    grid = grid or md_grid(cfg, round_)
    stats = stats if stats is not None else {}
    out = []
    for si, start in enumerate(system):
        for T, _ in grid:
            a = bare(start)
            a.calc = calc
            MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(seed + si))
            dyn = Langevin(a, cfg.get("dt", 1.0) * units.fs, temperature_K=T, friction=0.02,
                           rng=np.random.default_rng(seed + 1000 + si))
            frames = []
            dyn.attach(lambda: frames.append(bare(a)), interval=int(cfg["dump"]))
            ok = True
            try:
                dyn.run(int(cfg["steps"]))
            except Exception as exc:                      # an exploding trajectory still yields frames
                ok = False
                LOG(f"[warn]ASE MD at {T:.0f} K stopped: {type(exc).__name__}: {exc}[/warn]")
            if ok and frames and md_blew_up(frames[-1]):
                ok = False
                LOG(f"[warn]ASE MD at {T:.0f} K from start {si}: atoms collapsed (< 0.5 Å) — counted as failed[/warn]")
            stats["md_runs"] = stats.get("md_runs", 0) + 1
            stats["md_failed"] = stats.get("md_failed", 0) + (not ok)
            out += tag(frames, mflow_source=label, mflow_T=float(T), mflow_start=si, mflow_md_ok=ok)
    ase.io.write(workdir / f"{label}_ase.xyz", out, format="extxyz")
    LOG(f"{label} (ase): {len(out)} frames from {len(system)} start structure(s) × {len(grid)} conditions")
    return out


def explore_md_lammps(system: list, cfg: dict, deploy: str, round_: int, workdir: Path, seed: int,
                      lmp: str | None = None, grid: list | None = None, stats: dict | None = None,
                      plumed: str | None = None, label: str = "md") -> list:
    from ..analysis.traj import read_dump
    from ..lmp.run import cmd_md

    stats = stats if stats is not None else {}
    out = []
    for si, start in enumerate(system):
        start_file = workdir / f"start_{si}.xyz"
        ase.io.write(start_file, bare(start), format="extxyz")
        for T, P in grid or md_grid(cfg, round_):
            run_dir = workdir / f"{label}_s{si}_T{T:.0f}_P{P:g}"
            ens = "npt" if len(cfg.get("press", [])) > 1 or P != 1.0 else "nvt"
            args = Namespace(input=str(start_file), index="-1", deploy=deploy, ensemble=ens, temp=T,
                             temp_end=None, press=P, dt=cfg.get("dt", 1.0), steps=int(cfg["steps"]), tdamp=None,
                             pdamp=None, seed=seed + si, dump=int(cfg["dump"]), thermo=int(cfg["dump"]),
                             restart=max(int(cfg["steps"]), 1), ngpu=1, lmp=lmp, outdir=str(run_dir), skip=0.0,
                             nocheck=False, dry=False, resume=False, force=False, no_plot=True, plumed=plumed)
            code = cmd_md(args)
            stats["md_runs"] = stats.get("md_runs", 0) + 1
            stats["md_failed"] = stats.get("md_failed", 0) + (code != 0)
            dump = run_dir / "traj.lammpstrj"
            if not dump.exists():
                LOG(f"[warn]no trajectory from {run_dir.name} (exit {code})[/warn]")
                continue
            species = json.loads((run_dir / "md.json").read_text())["params"]["species"]
            frames = [bare(a) for a in read_dump(dump, 1, species)]
            for a in frames:
                a.wrap()
            out += tag(frames, mflow_source=label, mflow_T=float(T), mflow_P=float(P), mflow_start=si,
                       mflow_md_ok=bool(code == 0))
    LOG(f"{label} (lammps): {len(out)} frames")
    return out


# ============================================================================ #
#  heat-and-quench, random structure search, uncertainty-biased MD
# ============================================================================ #

def explore_heatquench(system: list, cfg: dict, model: str, workdir: Path, seed: int, calc=None,
                       stats: dict | None = None) -> list:
    """Langevin at T_high for steps_hot, then T ramps linearly to T_low over steps_quench."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    calc = calc or build_calculator(model, pick_device("auto"), "float64", None)
    stats = stats if stats is not None else {}
    T_hi, T_lo = float(cfg["T_high"]), float(cfg["T_low"])
    hot, quench, every = int(cfg["steps_hot"]), int(cfg["steps_quench"]), int(cfg["dump"])
    out = []
    for si, start in enumerate(system):
        a = bare(start)
        a.calc = calc
        MaxwellBoltzmannDistribution(a, temperature_K=T_hi, rng=np.random.default_rng(seed + si))
        dyn = Langevin(a, cfg.get("dt", 1.0) * units.fs, temperature_K=T_hi, friction=0.02,
                       rng=np.random.default_rng(seed + 2000 + si))
        frames, ok = [], True
        try:
            for done in range(0, hot + quench, every):
                end = done + every                               # the ramp reaches T_low on the last chunk
                T = T_hi if end <= hot else T_hi + (T_lo - T_hi) * min(1.0, (end - hot) / max(quench, 1))
                dyn.set_temperature(temperature_K=T)
                dyn.run(every)
                f = bare(a)
                f.info["mflow_T"] = float(T)
                frames.append(f)
        except Exception as exc:
            ok = False
            LOG(f"[warn]heat-quench from start {si} stopped: {type(exc).__name__}: {exc}[/warn]")
        if ok and frames and md_blew_up(frames[-1]):
            ok = False
        stats["md_runs"] = stats.get("md_runs", 0) + 1
        stats["md_failed"] = stats.get("md_failed", 0) + (not ok)
        out += [_keep(f, "mflow_T") for f in frames]
        tag(out[-len(frames):] if frames else [], mflow_source="heatquench", mflow_start=si, mflow_md_ok=ok)
    LOG(f"heatquench: {len(out)} frames ({T_hi:.0f} K → {T_lo:.0f} K)")
    return out


def random_cell(start: Atoms, rng, vol_jitter: float, min_dist: float, tries: int = 2000) -> Atoms | None:
    """Same composition, volume per atom × (1 ± jitter), random cell shape, no two atoms too close."""
    from ase.data import covalent_radii
    n = len(start)
    vol = start.get_volume() * (1.0 + rng.uniform(-vol_jitter, vol_jitter))
    lengths = rng.uniform(0.8, 1.25, 3)
    angles = np.radians(rng.uniform(75.0, 105.0, 3))
    from ase.geometry import cellpar_to_cell
    cell = cellpar_to_cell(np.concatenate([lengths, np.degrees(angles)]))
    cell *= (vol / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
    numbers = rng.permutation(start.numbers)
    a = Atoms(numbers=[], cell=cell, pbc=True)
    for z in numbers:
        for _ in range(tries):
            frac = rng.uniform(0, 1, 3)
            trial = a.copy()
            trial += Atoms(numbers=[z], scaled_positions=[frac], cell=cell, pbc=True)
            if len(trial) == 1:
                a = trial
                break
            d = trial.get_distances(len(trial) - 1, range(len(trial) - 1), mic=True)
            need = min_dist * (covalent_radii[z] + covalent_radii[trial.numbers[:-1]])
            if np.all(d > need):
                a = trial
                break
        else:
            return None
    return a if len(a) == n else None


def explore_rss(system: list, cfg: dict, model: str, workdir: Path, seed: int, calc=None) -> list:
    """AIRSS-lite: random cells relaxed with the model (FIRE on positions + cell)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    calc = calc or build_calculator(model, pick_device("auto"), "float64", None)
    rng = np.random.default_rng(seed + 77)
    out = []
    for si, start in enumerate(system):
        if not all(start.pbc):
            LOG(f"[warn]rss: start {si} is not periodic in 3D — skipped[/warn]")
            continue
        made = 0
        for k in range(int(cfg["n"])):
            for _ in range(20):                           # dense cells sometimes jam: draw a new cell
                a = random_cell(start, rng, float(cfg["vol_jitter"]), float(cfg["min_dist"]))
                if a is not None:
                    break
            if a is None:
                LOG(f"[warn]rss: no random cell with min_dist {cfg['min_dist']} after 20 draws — lower it[/warn]")
                continue
            made += 1
            a.calc = calc
            path = []
            try:
                opt = FIRE(FrechetCellFilter(a), logfile=None)
                opt.attach(lambda: path.append(bare(a)), interval=int(cfg.get("every", 5)))
                opt.run(fmax=float(cfg["fmax"]), steps=int(cfg["steps"]))
                end = bare(a)
                end.info["mflow_E_explore"] = float(a.get_potential_energy())
                path.append(end)
            except Exception as exc:
                LOG(f"[warn]rss {si}/{k}: relaxation stopped: {type(exc).__name__}: {exc}[/warn]")
            out += tag(path, mflow_source="rss", mflow_start=si)
        LOG(f"rss start {si}: {made} random cells relaxed")
    LOG(f"rss: {len(out)} frames (relaxation paths + end points)")
    return out


def explore_ubmd(system: list, cfg: dict, models: list, workdir: Path, seed: int, calcs: list | None = None,
                 stats: dict | None = None) -> list:
    from .bias import BiasedCommitteeCalculator
    if calcs is None:
        if len(models) < 2:
            LOG("[warn]ubmd needs a committee of at least 2 models — skipped (train.committee ≥ 2)[/warn]")
            return []
        calcs = [build_calculator(m, pick_device("auto"), "float64", None) for m in models]
    biased = BiasedCommitteeCalculator(calcs, kappa=float(cfg["kappa"]))
    grid = [(float(T), 1.0) for T in cfg["temps"]]
    frames = explore_md_ase(system, cfg, "", 0, workdir, seed + 500, grid=grid, calc=biased, stats=stats,
                            label="ubmd")
    LOG("[dim]ubmd: biased towards committee disagreement — frames may be unphysical; the failed threshold "
        "and mflow check filter those[/dim]")
    return frames


# ============================================================================ #
#  minima hopping
# ============================================================================ #

def explore_minhop(system: list, cfg: dict, model: str, workdir: Path, calc=None) -> list:
    """
    ASE's MinimaHopping (fixed cell). It writes its files into the working directory, so each
    start structure gets its own. Every local minimum it reaches is kept (qn*.traj, last frame),
    not only the accepted ones in minima.traj: a rejected minimum is just as new to the model.
    Plus every 10th frame of the MD escapes between them.
    """
    from ase.optimize.minimahopping import MinimaHopping

    calc = calc or build_calculator(model, pick_device("auto"), "float64", None)
    out = []
    cwd = os.getcwd()
    for si, start in enumerate(system):
        d = workdir / f"minhop_{si}"
        d.mkdir(parents=True, exist_ok=True)
        os.chdir(d)
        minima, md = [], []
        try:
            a = bare(start)
            a.calc = calc
            hop = MinimaHopping(a, T0=float(cfg.get("T0", 1000.0)), fmax=float(cfg.get("fmax", 0.05)))
            hop(totalsteps=int(cfg.get("steps", 20)))
        except Exception as exc:
            LOG(f"[warn]minima hopping from start {si} stopped: {type(exc).__name__}: {exc}[/warn]")
        finally:
            try:
                for f in sorted(Path(".").glob("qn*.traj")):
                    m = ase.io.read(f, -1)
                    m.info["mflow_E_explore"] = float(m.get_potential_energy())
                    minima.append(m)
                for f in sorted(Path(".").glob("md*.traj")):
                    md += ase.io.read(f, "::10")
            finally:
                os.chdir(cwd)
        energies = [m.info["mflow_E_explore"] for m in minima]
        distinct = len({round(e, 3) for e in energies})
        out += tag([_keep(m, "mflow_E_explore") for m in minima], mflow_source="minhop_min", mflow_start=si)
        out += tag([bare(m) for m in md], mflow_source="minhop_md", mflow_start=si)
        if energies:
            LOG(f"minhop start {si}: {len(minima)} local minima ({distinct} distinct energies, "
                f"lowest {min(energies):.3f} eV), {len(md)} MD frames")
    LOG(f"minhop: {len(out)} structures (minima + MD between them)")
    return out


def _keep(atoms: Atoms, *keys) -> Atoms:
    out = bare(atoms)
    out.info.update({k: atoms.info[k] for k in keys if k in atoms.info})
    return out


# ============================================================================ #
#  driver
# ============================================================================ #

def drivers(camp) -> list[str]:
    """Models that drive the MD: member 0 by default, explore.md.drivers members (or all) on request."""
    models = camp.committee() or [camp.settings["foundation"]]
    want = camp.settings["explore"]["md"].get("drivers", 1)
    n = len(models) if str(want) == "all" else max(1, min(int(want), len(models)))
    return models[:n]


def advance_region(camp) -> dict:
    """Leave every region that already passed L1 (in order); returns the region to explore now."""
    from .convergence import current_region, l1_passed, regions
    regs = regions(camp)
    if len(regs) == 1:
        return regs[0]
    led = camp.ledger()
    i = int(camp.state.get("region_index", 0))
    while i < len(regs) - 1 and l1_passed(camp, led, regs[i]["name"]):
        LOG(f"[ok]region {regs[i]['name']} passed[/ok] → exploring {regs[i + 1]['name']}")
        camp.log("region", int(camp.state.get("round", 0)), passed=regs[i]["name"], next=regs[i + 1]["name"])
        i += 1
        camp.save_state(region_index=i)
    return current_region(camp)


def run_explorers(camp, round_: int, lmp: str | None = None) -> tuple[list, dict]:
    from .convergence import regions
    s = camp.settings
    system = ase.io.read(camp.data / "system.xyz", ":")
    workdir = camp.round_dir(round_, "explore")
    model = camp.explorer_model()
    seed = 1000 * round_
    frames, counts = [], {}
    ex = s["explore"]
    regs = regions(camp)
    region = advance_region(camp)
    if len(regs) > 1:
        k = sum(1 for x in camp.ledger() if x["step"] == "explore" and x.get("region") == region["name"])
        md_cfg, grid = region_cfg(ex["md"], region, k)
        todo = [(region, md_cfg, grid)]
        if ex["md"].get("keep_passed"):
            i = int(camp.state.get("region_index", 0))
            todo = [(r, *region_cfg(ex["md"], r, k)) for r in regs[:i]] + todo
        LOG(f"region [hi]{region['name']}[/hi] (round {k + 1} in it): T {', '.join(f'{T:.0f}' for T, _ in grid)} K · "
            f"{md_cfg['steps']} steps")
    else:
        todo = [(region, ex["md"], md_grid(ex["md"], round_))]
    counts["region"] = region["name"]
    if ex["perturb"].get("enabled"):
        got = explore_perturb(system, ex["perturb"], seed)
        frames += got
        counts["perturb"] = len(got)
    if ex["md"].get("enabled"):
        md_stats = {}
        got = []
        for d, drv in enumerate(drivers(camp)):
            for reg, md_cfg, grid in todo:
                sub = workdir if d == 0 else workdir / f"driver{d}"
                sub.mkdir(parents=True, exist_ok=True)
                if md_cfg.get("engine") == "lammps":
                    deploy = md_cfg.get("deploy") if d == 0 else None
                    if deploy in (None, "auto"):
                        deploy = auto_deploy(camp, round_, drv, tag_=f"deploy{d or ''}")
                    part = explore_md_lammps(system, md_cfg, deploy, round_, sub, seed + 100 * d, lmp, grid=grid,
                                             stats=md_stats)
                else:
                    part = explore_md_ase(system, md_cfg, drv, round_, sub, seed + 100 * d, grid=grid, stats=md_stats)
                got += tag(part, mflow_region=reg["name"], mflow_driver=d)
        frames += got
        counts.update({"md": len(got), **md_stats})
    if ex["minhop"].get("enabled"):
        got = explore_minhop(system, ex["minhop"], model, workdir)
        frames += got
        counts["minhop"] = len(got)
    hq = ex.get("heatquench") or {}
    if hq.get("enabled"):
        st = {}
        got = explore_heatquench(system, hq, model, workdir, seed, stats=st)
        frames += got
        counts["heatquench"] = len(got)
        counts["md_runs"] = counts.get("md_runs", 0) + st.get("md_runs", 0)
        counts["md_failed"] = counts.get("md_failed", 0) + st.get("md_failed", 0)
    rss = ex.get("rss") or {}
    if rss.get("enabled"):
        got = explore_rss(system, rss, model, workdir, seed)
        frames += got
        counts["rss"] = len(got)
    ub = ex.get("ubmd") or {}
    if ub.get("enabled"):
        got = explore_ubmd(system, ub, camp.committee(), workdir, seed)
        frames += got
        counts["ubmd"] = len(got)
    pl = ex.get("plumed") or {}
    if pl.get("enabled"):
        if not pl.get("file"):
            raise SystemExit("explore.plumed.enabled needs explore.plumed.file (a PLUMED input)")
        deploy = ex["md"].get("deploy")
        if deploy in (None, "auto"):
            deploy = auto_deploy(camp, round_, model)
        st = {}
        got = explore_md_lammps(system, {**ex["md"], **pl}, deploy, round_, workdir, seed + 900, lmp,
                                grid=[(float(T), 1.0) for T in pl["temps"]], stats=st,
                                plumed=str(Path(pl["file"]).expanduser().resolve()), label="plumed")
        frames += got
        counts["plumed"] = len(got)
    for a in frames:
        a.info.setdefault("mflow_region", region["name"])
    ase.io.write(workdir / "explored.xyz", frames, format="extxyz")
    return frames, counts


def auto_deploy(camp, round_: int, model: str, tag_: str = "deploy") -> str:
    """Export the explorer model for LAMMPS once per round (GPU needed)."""
    from ..lmp.deploy import cmd_deploy
    out = camp.round_dir(round_, tag_)
    if not (out / "deploy.json").exists():
        code = cmd_deploy(Namespace(model=model, head=None, dtype="float32", outdir=str(out)))
        if code != 0:
            raise RuntimeError(f"deploy of {model} failed — set explore.md.deploy or engine: ase")
    return str(out / "deploy.json")
