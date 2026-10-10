# -*- coding: utf-8 -*-
"""
LAMMPS input generation — one fixed template, every choice visible in the file.

Units are LAMMPS `metal`: Å, eV, ps, bar. MFlow's own options are in fs (dt, tdamp, pdamp)
and converted here. Element ↔ type mapping is decided once (`species_for`) and used for the
data file, pair_coeff and dump_modify alike.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import ase.io
from ase.data import atomic_numbers

from .. import __version__
from ..thresholds import section

ENSEMBLES = ("nvt", "npt", "nve")
HALT_FACTOR = section("md")["halt_factor"]   # stop when T exceeds this × the target (thresholds.yaml)
FS = 1e-3                  # fs -> ps


@dataclass
class MDParams:
    ensemble: str = "nvt"
    temp: float = 300.0
    temp_end: float | None = None
    press: float = 1.0               # bar
    dt: float = 1.0                  # fs
    steps: int = 100_000
    tdamp: float | None = None       # fs, default 100 dt
    pdamp: float | None = None       # fs, default 1000 dt
    seed: int = 42
    dump: int = 100
    thermo: int = 100
    restart: int = 10_000
    species: list = field(default_factory=list)
    pair_style: str = ""
    pair_coeff: str = "* * {species}"   # {species} -> the element names in type order
    pair_extra: list = field(default_factory=list)   # e.g. ["pair_modify shift yes"] (tests)
    periodic: tuple = (True, True, True)

    def __post_init__(self):
        if self.ensemble not in ENSEMBLES:
            raise ValueError(f"ensemble must be one of {ENSEMBLES}")
        self.temp_end = self.temp if self.temp_end is None else self.temp_end
        self.tdamp = 100 * self.dt if self.tdamp is None else self.tdamp
        self.pdamp = 1000 * self.dt if self.pdamp is None else self.pdamp

    def to_dict(self) -> dict:
        return asdict(self)


def species_for(atoms_list, model_elements: list) -> list:
    """Elements present, ordered by atomic number; each must be known to the model."""
    present = sorted({s for a in atoms_list for s in a.get_chemical_symbols()}, key=lambda s: atomic_numbers[s])
    unknown = [s for s in present if s not in model_elements]
    if unknown:
        raise ValueError(f"the model has no parameters for {' '.join(unknown)} "
                         f"(model elements: {' '.join(model_elements)})")
    return present


def write_data(atoms, path: Path, species: list):
    """atom_style atomic, masses included, types in `species` order (LAMMPS rotates the cell)."""
    ase.io.write(path, atoms, format="lammps-data", specorder=species, masses=True,
                 atom_style="atomic", units="metal")


def read_back(path: Path, species: list):
    """The structure exactly as LAMMPS sees it — rotated cell, same atom order."""
    atoms = ase.io.read(path, format="lammps-data", atom_style="atomic", units="metal")
    atoms.set_chemical_symbols([species[t - 1] for t in atoms.arrays["type"]])
    return atoms


def boundary(periodic) -> str:
    return " ".join("p" if p else "f" for p in periodic)


def header(p: MDParams, kind: str) -> list[str]:
    return [f"# MFlow {__version__} · {kind} · written {datetime.now().isoformat(timespec='seconds')}",
            "# units metal: Å, eV, ps, bar",
            ""]


def common_setup(p: MDParams, data_cmd: str) -> list[str]:
    lines = ["units           metal",
             "atom_style      atomic",
             "atom_modify     map yes",
             "newton          on",
             f"boundary        {boundary(p.periodic)}",
             data_cmd,
             "",
             f"pair_style      {p.pair_style}",
             f"pair_coeff      {p.pair_coeff.format(species=' '.join(p.species))}"]
    lines += list(p.pair_extra)
    lines += ["",
              "neighbor        2.0 bin",
              "neigh_modify    every 1 delay 0 check yes"]
    return lines


def ensemble_fix(p: MDParams) -> str:
    t0, t1, td = p.temp, p.temp_end, p.tdamp * FS
    if p.ensemble == "nvt":
        return f"fix             md all nvt temp {t0} {t1} {td:g}"
    if p.ensemble == "npt":
        return f"fix             md all npt temp {t0} {t1} {td:g} iso {p.press} {p.press} {p.pdamp * FS:g}"
    return "fix             md all nve"


def md_input(p: MDParams, data_file: str = "data.lmp", resume_from: str | None = None,
             log_file: str = "log.lammps") -> str:
    """The production input. `resume_from` swaps read_data/velocity for read_restart."""
    halt = HALT_FACTOR * max(p.temp, p.temp_end, 1.0)
    data_cmd = f"read_restart    {resume_from}" if resume_from else f"read_data       {data_file}"
    lines = header(p, f"{p.ensemble} MD" + (" (resumed)" if resume_from else ""))
    lines += [f"log             {log_file} append", ""]
    lines += common_setup(p, data_cmd)
    lines += ["",
              f"timestep        {p.dt * FS:g}",
              "thermo_style    custom step time temp pe ke etotal press vol density pxx pyy pzz",
              "thermo_modify   lost error flush yes",
              f"thermo          {p.thermo}",
              ""]
    if not resume_from:
        lines += [f"velocity        all create {p.temp} {p.seed} dist gaussian mom yes rot yes", ""]
    lines += [ensemble_fix(p),
              "variable        tnow equal temp",
              f"fix             runaway all halt {p.thermo} v_tnow > {halt:g} error soft",
              "",
              f"dump            traj all custom {p.dump} traj.lammpstrj id type element xu yu zu vx vy vz fx fy fz",
              f"dump_modify     traj element {' '.join(p.species)} sort id" + (" append yes" if resume_from else ""),
              f"restart         {p.restart} restart.a restart.b",
              "",
              f"run             {p.steps} upto",
              "write_restart   final.restart",
              ""]
    return "\n".join(lines)


def check_input(p: MDParams, data_file: str = "data.lmp") -> str:
    """run 0 on the start structure: energy, pressure and per-atom forces for the consistency check."""
    lines = header(p, "consistency check (run 0)")
    lines += ["log             check.log", ""]
    lines += common_setup(p, f"read_data       {data_file}")
    lines += ["",
              "thermo_style    custom step pe press pxx pyy pzz pxy pxz pyz vol",
              "thermo_modify   format float %20.12g",
              "run             0",
              "write_dump      all custom check_forces.dump id type x y z fx fy fz modify sort id "
              "format float %20.12g",
              ""]
    return "\n".join(lines)


def sanity(p: MDParams, atoms, r_max: float | None, ngpu: int, rep) -> None:
    """Parameter rules — warnings, not hard stops (except where physics forbids)."""
    has_h = "H" in atoms.get_chemical_symbols()
    dt_max = 1.0 if has_h else 2.0
    rep.check("timestep", "warn" if p.dt > dt_max else "pass", f"{p.dt} fs" + (" (H present)" if has_h else ""),
              suggest=f"timestep above {dt_max} fs{' with H' if has_h else ''} risks energy drift: lower -dt")
    if p.ensemble in ("nvt", "npt"):
        ratio = p.tdamp / p.dt
        rep.check("thermostat damping", "warn" if not 20 <= ratio <= 1000 else "pass",
                  f"tdamp {p.tdamp:g} fs = {ratio:.0f} dt",
                  suggest="Nosé–Hoover tdamp is usually ~100 timesteps")
    if p.ensemble == "npt":
        if not all(atoms.pbc):
            rep.check("npt boundary", "fail", "NPT needs a fully periodic cell")
        ratio = p.pdamp / p.dt
        rep.check("barostat damping", "warn" if not 200 <= ratio <= 10000 else "pass",
                  f"pdamp {p.pdamp:g} fs = {ratio:.0f} dt", suggest="barostat pdamp is usually ~1000 timesteps")
    n = len(atoms)
    if ngpu:
        rep.check("atoms per GPU", "warn" if n / max(ngpu, 1) < 500 else "pass", f"{n / max(ngpu, 1):.0f}",
                  suggest="fewer than ~500 atoms per GPU leaves the GPU mostly idle (1000–4000 is the sweet spot)")
    if r_max and any(atoms.pbc):
        widths = [w for w, per in zip(cell_widths(atoms), atoms.pbc) if per]
        thin = min(widths) < 2 * r_max
        rep.check("cell size", "warn" if thin else "pass",
                  f"thinnest periodic width {min(widths):.2f} Å vs 2·r_max {2 * r_max:.2f} Å",
                  suggest="cell narrower than 2·r_max: atoms interact with their own periodic images "
                          "(finite-size effects) — use a supercell for production MD")
    rep.check("dump interval", "warn" if p.steps // max(p.dump, 1) > 100_000 else "pass",
              f"{p.steps // max(p.dump, 1)} frames", suggest="more than 1e5 dump frames: raise -dump")


def cell_widths(atoms) -> np.ndarray:
    """Distance between opposite faces of the cell, per direction."""
    cell = np.array(atoms.cell)
    vol = abs(np.linalg.det(cell))
    return np.array([vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])
