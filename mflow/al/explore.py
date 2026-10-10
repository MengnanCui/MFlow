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


def explore_md_ase(system: list, cfg: dict, model: str, round_: int, workdir: Path, seed: int) -> list:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    calc = build_calculator(model, pick_device("auto"), "float64", None)
    out = []
    for si, start in enumerate(system):
        for T, _ in md_grid(cfg, round_):
            a = bare(start)
            a.calc = calc
            MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(seed + si))
            dyn = Langevin(a, cfg.get("dt", 1.0) * units.fs, temperature_K=T, friction=0.02,
                           rng=np.random.default_rng(seed + 1000 + si))
            frames = []
            dyn.attach(lambda: frames.append(bare(a)), interval=int(cfg["dump"]))
            try:
                dyn.run(int(cfg["steps"]))
            except Exception as exc:                      # an exploding trajectory still yields frames
                LOG(f"[warn]ASE MD at {T:.0f} K stopped: {type(exc).__name__}: {exc}[/warn]")
            out += tag(frames, mflow_source="md", mflow_T=float(T), mflow_start=si)
    ase.io.write(workdir / "md_ase.xyz", out, format="extxyz")
    LOG(f"md (ase): {len(out)} frames from {len(system)} start structure(s) × {len(md_grid(cfg, round_))} conditions")
    return out


def explore_md_lammps(system: list, cfg: dict, deploy: str, round_: int, workdir: Path, seed: int,
                      lmp: str | None = None) -> list:
    from ..analysis.traj import read_dump
    from ..lmp.run import cmd_md

    out = []
    for si, start in enumerate(system):
        start_file = workdir / f"start_{si}.xyz"
        ase.io.write(start_file, bare(start), format="extxyz")
        for T, P in md_grid(cfg, round_):
            run_dir = workdir / f"md_s{si}_T{T:.0f}_P{P:g}"
            ens = "npt" if len(cfg.get("press", [])) > 1 or P != 1.0 else "nvt"
            args = Namespace(input=str(start_file), index="-1", deploy=deploy, ensemble=ens, temp=T,
                             temp_end=None, press=P, dt=cfg.get("dt", 1.0), steps=int(cfg["steps"]), tdamp=None,
                             pdamp=None, seed=seed + si, dump=int(cfg["dump"]), thermo=int(cfg["dump"]),
                             restart=max(int(cfg["steps"]), 1), ngpu=1, lmp=lmp, outdir=str(run_dir), skip=0.0,
                             nocheck=False, dry=False, resume=False, force=False, no_plot=True)
            code = cmd_md(args)
            dump = run_dir / "traj.lammpstrj"
            if not dump.exists():
                LOG(f"[warn]no trajectory from {run_dir.name} (exit {code})[/warn]")
                continue
            species = json.loads((run_dir / "md.json").read_text())["params"]["species"]
            frames = [bare(a) for a in read_dump(dump, 1, species)]
            for a in frames:
                a.wrap()
            out += tag(frames, mflow_source="md", mflow_T=float(T), mflow_P=float(P), mflow_start=si,
                       mflow_md_ok=bool(code == 0))
    LOG(f"md (lammps): {len(out)} frames")
    return out


# ============================================================================ #
#  minima hopping
# ============================================================================ #

def explore_minhop(system: list, cfg: dict, model: str, workdir: Path) -> list:
    """ASE's MinimaHopping (fixed cell). Writes its files in its own directory, so each start gets one."""
    from ase.optimize.minimahopping import MinimaHopping

    calc = build_calculator(model, pick_device("auto"), "float64", None)
    out = []
    cwd = os.getcwd()
    for si, start in enumerate(system):
        d = workdir / f"minhop_{si}"
        d.mkdir(parents=True, exist_ok=True)
        os.chdir(d)
        try:
            a = bare(start)
            a.calc = calc
            hop = MinimaHopping(a, T0=float(cfg.get("T0", 1000.0)), fmax=float(cfg.get("fmax", 0.05)))
            hop(totalsteps=int(cfg.get("steps", 20)))
            minima = ase.io.read("minima.traj", ":") if Path("minima.traj").exists() else []
            md = []
            for f in sorted(Path(".").glob("md*.traj")):
                md += ase.io.read(f, "::10")
        except Exception as exc:
            LOG(f"[warn]minima hopping from start {si} stopped: {type(exc).__name__}: {exc}[/warn]")
            minima, md = [], []
        finally:
            os.chdir(cwd)
        out += tag([bare(m) for m in minima], mflow_source="minhop_min", mflow_start=si)
        out += tag([bare(m) for m in md], mflow_source="minhop_md", mflow_start=si)
    LOG(f"minhop: {len(out)} structures (minima + MD between them)")
    return out


# ============================================================================ #
#  driver
# ============================================================================ #

def run_explorers(camp, round_: int, lmp: str | None = None) -> tuple[list, dict]:
    s = camp.settings
    system = ase.io.read(camp.data / "system.xyz", ":")
    workdir = camp.round_dir(round_, "explore")
    model = camp.explorer_model()
    seed = 1000 * round_
    frames, counts = [], {}
    ex = s["explore"]
    if ex["perturb"].get("enabled"):
        got = explore_perturb(system, ex["perturb"], seed)
        frames += got
        counts["perturb"] = len(got)
    if ex["md"].get("enabled"):
        if ex["md"].get("engine") == "lammps":
            deploy = ex["md"].get("deploy")
            if deploy in (None, "auto"):
                deploy = auto_deploy(camp, round_, model)
            got = explore_md_lammps(system, ex["md"], deploy, round_, workdir, seed, lmp)
        else:
            got = explore_md_ase(system, ex["md"], model, round_, workdir, seed)
        frames += got
        counts["md"] = len(got)
    if ex["minhop"].get("enabled"):
        got = explore_minhop(system, ex["minhop"], model, workdir)
        frames += got
        counts["minhop"] = len(got)
    ase.io.write(workdir / "explored.xyz", frames, format="extxyz")
    return frames, counts


def auto_deploy(camp, round_: int, model: str) -> str:
    """Export the explorer model for LAMMPS once per round (GPU needed)."""
    from ..lmp.deploy import cmd_deploy
    out = camp.round_dir(round_, "deploy")
    if not (out / "deploy.json").exists():
        code = cmd_deploy(Namespace(model=model, head=None, dtype="float32", outdir=str(out)))
        if code != 0:
            raise RuntimeError(f"deploy of {model} failed — set explore.md.deploy or engine: ase")
    return str(out / "deploy.json")
