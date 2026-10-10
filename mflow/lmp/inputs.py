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
THERMOSTATS = ("nh", "csvr", "langevin")
BAROSTATS = ("iso", "aniso", "tri")
PROTOCOLS = ("single", "equilibrate", "transport", "anneal", "quench", "minimize", "melt")
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
    thermostat: str = "nh"           # nh (Nosé–Hoover chain) | csvr (Bussi) | langevin
    baro: str = "iso"                # iso | aniso | tri (tri needs a triclinic box: written with tilts)
    protocol: str = "single"
    melt_temp: float | None = None   # melt: temperature of the molten half [1.5 × temp]
    fmax: float = 0.01               # minimize: max force component, eV/Å
    shell_cutoff: float | None = None  # melt: first-shell radius for the solid-bond order parameter, Å
    triclinic: bool = False          # the data file has tilt factors

    def __post_init__(self):
        if self.ensemble not in ENSEMBLES:
            raise ValueError(f"ensemble must be one of {ENSEMBLES}")
        if self.thermostat not in THERMOSTATS or self.baro not in BAROSTATS or self.protocol not in PROTOCOLS:
            raise ValueError(f"thermostat {THERMOSTATS}, baro {BAROSTATS}, protocol {PROTOCOLS}")
        self.temp_end = self.temp if self.temp_end is None else self.temp_end
        self.tdamp = 100 * self.dt if self.tdamp is None else self.tdamp
        self.pdamp = 1000 * self.dt if self.pdamp is None else self.pdamp
        self.melt_temp = 1.5 * self.temp if self.melt_temp is None else self.melt_temp
        self.triclinic = self.triclinic or self.baro == "tri"

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================ #
#  protocols: a list of stages, each one `run N upto`
# ============================================================================ #

@dataclass
class Stage:
    name: str
    ensemble: str                   # nvt | npt | nph | nve | min
    steps: int
    temp: float = 300.0
    temp_end: float = 300.0
    group: str = "all"              # atoms integrated in this stage
    rescale: bool = False           # npt: set the volume to its average over the second half afterwards
    start: int = 0                  # filled by stages()
    end: int = 0


def stages(p: MDParams) -> list[Stage]:
    """
    single       one stage, -ensemble
    equilibrate  NPT (volume then fixed at its average) → NVT; NVT only without full periodicity
    transport    NPT (average volume) 20 % → NVT 10 % → NVE 70 %: diffusion from a run with no
                 thermostat acting on the dynamics
    anneal       heat temp → temp_end 40 %, hold 20 %, cool back 40 % (NPT if -ensemble npt)
    quench       hold at temp 20 %, then cool temp → temp_end 80 %
    minimize     energy minimisation, cell relaxed with box/relax when periodic (-steps = max iterations)
    melt         solid-liquid coexistence: NPT crystal 20 % → melt the upper half (z) at melt_temp, NVT, lower
                 half frozen 15 % → both halves set to temp → NPH all 65 %, x-y coupled, z free; T drifts
                 to T_m while both phases survive (solid-bond count per slab tells)
    """
    n, T0, T1 = p.steps, p.temp, p.temp_end
    full = all(p.periodic)
    nvx = "npt" if p.ensemble == "npt" and full else "nvt"
    split = lambda *fr: [int(round(n * f)) for f in fr[:-1]] + [n - sum(int(round(n * f)) for f in fr[:-1])]
    if p.protocol == "single":
        out = [Stage("production", p.ensemble, n, T0, T1)]
    elif p.protocol == "equilibrate":
        if full:
            a, b = split(0.5, 0.5)
            out = [Stage("npt", "npt", a, T0, T0, rescale=True), Stage("nvt", "nvt", b, T0, T0)]
        else:
            out = [Stage("nvt", "nvt", n, T0, T0)]
    elif p.protocol == "transport":
        if full:
            a, b, c = split(0.2, 0.1, 0.7)
            out = [Stage("npt", "npt", a, T0, T0, rescale=True), Stage("nvt", "nvt", b, T0, T0),
                   Stage("nve", "nve", c, T0, T0)]
        else:
            a, c = split(0.3, 0.7)
            out = [Stage("nvt", "nvt", a, T0, T0), Stage("nve", "nve", c, T0, T0)]
    elif p.protocol == "anneal":
        a, b, c = split(0.4, 0.2, 0.4)
        out = [Stage("heat", nvx, a, T0, T1), Stage("hold", nvx, b, T1, T1), Stage("cool", nvx, c, T1, T0)]
    elif p.protocol == "quench":
        a, b = split(0.2, 0.8)
        out = [Stage("hold", nvx, a, T0, T0), Stage("cool", nvx, b, T0, T1)]
    elif p.protocol == "minimize":
        out = [Stage("minimize", "min", n, 0.0, 0.0)]
    else:                                                       # melt
        a, b, d = split(0.2, 0.15, 0.65)
        Tm = p.melt_temp
        out = [Stage("solid", "npt", a, T0, T0), Stage("melt", "nvt", b, Tm, Tm, group="liquid"),
               Stage("coexistence", "nph", d, T0, T0)]
    step = 0
    for st in out:
        st.start, st.end = step, step + st.steps
        step = st.end
    return out


def baro_spec(p: MDParams, coexist: bool = False) -> str:
    P, pd = p.press, p.pdamp * FS
    if coexist:                    # interface normal z: x and y coupled, z on its own (more stable than aniso)
        return f"x {P} {P} {pd:g} y {P} {P} {pd:g} z {P} {P} {pd:g} couple xy"
    return f"{p.baro} {P} {P} {pd:g}"


def integrator(p: MDParams, st: Stage) -> list[str]:
    """Fix lines for one stage. csvr / langevin only thermostat: positions come from nve / nph."""
    g, T0, T1, td = st.group, st.temp, st.temp_end, p.tdamp * FS
    seed = p.seed + st.start + 1
    if st.ensemble == "nve":
        return [f"fix             md {g} nve"]
    if st.ensemble == "nph":
        return [f"fix             md {g} nph {baro_spec(p, coexist=p.protocol == 'melt')}"]
    if p.thermostat == "nh":
        if st.ensemble == "nvt":
            return [f"fix             md {g} nvt temp {T0} {T1} {td:g}"]
        return [f"fix             md {g} npt temp {T0} {T1} {td:g} {baro_spec(p)}"]
    base = f"fix             md {g} nve" if st.ensemble == "nvt" else f"fix             md {g} nph {baro_spec(p)}"
    if p.thermostat == "csvr":
        return [base, f"fix             tstat {g} temp/csvr {T0} {T1} {td:g} {seed}"]
    return [base, f"fix             tstat {g} langevin {T0} {T1} {td:g} {seed} zero yes"]


def stage_lines(p: MDParams, st: Stage, resume_step: int | None) -> list[str]:
    lines = ["", f"# ---- stage {st.name}: {st.ensemble} {st.temp:g} -> {st.temp_end:g} K, steps {st.start}-{st.end}"]
    resumed_inside = resume_step is not None and resume_step > st.start
    if p.protocol == "melt" and st.name == "melt" and not resumed_inside:
        lines += ["variable        zmid equal zlo+0.5*lz",
                  "region          upper block INF INF INF INF ${zmid} INF units box",
                  "group           liquid region upper",
                  "group           solid subtract all liquid",
                  "velocity        solid set 0.0 0.0 0.0"]
    if p.protocol == "melt" and st.group == "liquid":
        lines += ["fix             freeze solid setforce 0.0 0.0 0.0"]
    if p.protocol == "melt" and st.name == "coexistence":
        if not resumed_inside:       # both halves start at T; NPH then trades latent heat against T
            lines += [f"velocity        liquid scale {st.temp}",
                      f"velocity        solid create {st.temp} {p.seed + 7} dist gaussian"]
        nf = max(100, (st.steps // 50) // 10 * 10)
        # ten Wolde solid bonds: first-shell neighbours j with normalised q6(i)·q6(j) > 0.5. A hot crystal
        # keeps most of them, a liquid few; raw per-atom q6 cannot tell a hot crystal from a liquid
        lines += ["compute         pea all pe/atom",
                  f"compute         q6 all orientorder/atom degrees 1 6 nnn NULL cutoff {p.shell_cutoff or 3.5:.3f} "
                  "components 6",
                  "compute         xbonds all coord/atom orientorder q6 0.5",
                  "compute         slabs all chunk/atom bin/1d z lower 0.05 units reduced",
                  f"fix             profile all ave/chunk 10 {nf // 10} {nf} slabs c_pea c_xbonds "
                  "file phase_profile.dat"]
    lines += integrator(p, st)
    rescale = st.rescale and st.start == 0 and not (resumed_inside and resume_step > st.start + st.steps // 2)
    if rescale:
        half = max(st.steps // 2, 10) // 10 * 10
        lines += ["variable        vnow equal vol",
                  f"fix             vave all ave/time 10 {max(half // 10, 1)} {st.end} v_vnow"]
    lines += [f"run             {st.end} upto"]
    if rescale:
        tilt = " xy final $(xy*v_s) xz final $(xz*v_s) yz final $(yz*v_s)" if p.triclinic else ""
        lines += ["variable        s equal (f_vave/vol)^(1.0/3.0)",
                  f"print           \"MFLOW volume -> average of the second half: scale ${{s}}\"",
                  f"change_box      all x scale ${{s}} y scale ${{s}} z scale ${{s}}{tilt} remap",
                  "unfix           vave"]
    lines += ["unfix           md"] + (["unfix           tstat"] if any(" tstat " in l for l in lines) else [])
    if p.protocol == "melt" and st.group == "liquid":
        lines += ["unfix           freeze"]
    return lines


def minimize_lines(p: MDParams) -> list[str]:
    """box/relax has no KOKKOS variant and cg/kk does not support it: run the minimiser unsuffixed."""
    lines = ["", "# ---- minimize: repeat until the cell stops changing (box/relax is not a well-defined",
             "#      minimisation, LAMMPS docs fix box/relax)",
             "suffix          off",
             "min_style       cg",
             "min_modify      line quadratic norm max"]
    if all(p.periodic):
        lines += [f"fix             relax all box/relax {p.baro} {p.press} vmax 0.001 nreset 50"]
    lines += ["variable        it loop 20",
              "label           relax_loop",
              "variable        v0 equal $(vol)",
              f"minimize        0.0 {p.fmax:g} {p.steps} {10 * p.steps}",
              'print           "MFLOW relax iteration ${it}: volume $(vol) pe $(pe)"',
              'if              "$(abs(vol-v_v0)/v_v0) < 1.0e-6" then "jump SELF relax_done"',
              "next            it",
              "jump            SELF relax_loop",
              "label           relax_done"]
    if all(p.periodic):
        lines += ["unfix           relax"]
    lines += ["write_data      relaxed.data",
              "write_dump      all custom relaxed.dump id type element x y z fx fy fz modify sort id "
              f"element {' '.join(p.species)} format float %20.12g"]
    return lines


def species_for(atoms_list, model_elements: list) -> list:
    """Elements present, ordered by atomic number; each must be known to the model."""
    present = sorted({s for a in atoms_list for s in a.get_chemical_symbols()}, key=lambda s: atomic_numbers[s])
    unknown = [s for s in present if s not in model_elements]
    if unknown:
        raise ValueError(f"the model has no parameters for {' '.join(unknown)} "
                         f"(model elements: {' '.join(model_elements)})")
    return present


def write_data(atoms, path: Path, species: list, triclinic: bool = False):
    """atom_style atomic, masses included, types in `species` order (LAMMPS rotates the cell).
    triclinic: write tilt factors even for an orthogonal cell (fix npt ... tri needs them)."""
    ase.io.write(path, atoms, format="lammps-data", specorder=species, masses=True,
                 atom_style="atomic", units="metal", force_skew=triclinic)


def first_shell(atoms) -> float:
    """1.2 × the median nearest-neighbour distance: between the 1st and 2nd shells of fcc/hcp, both bcc shells."""
    from ase.neighborlist import neighbor_list
    i, d = neighbor_list("id", atoms, 6.0)
    if not len(d):
        return 3.5
    nn = np.full(len(atoms), np.inf)
    np.minimum.at(nn, i, d)
    return float(1.2 * np.median(nn[np.isfinite(nn)]))


def is_orthogonal(atoms) -> bool:
    cell = np.array(atoms.cell)
    return bool(np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-8))


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


def md_input(p: MDParams, data_file: str = "data.lmp", resume_from: str | None = None,
             log_file: str = "log.lammps", resume_step: int | None = None) -> str:
    """The production input. `resume_from` swaps read_data/velocity for read_restart and drops
    the stages already finished before `resume_step`."""
    plan = stages(p)
    halt = HALT_FACTOR * max(p.temp, p.temp_end, p.melt_temp if p.protocol == "melt" else 0.0, 1.0)
    data_cmd = f"read_restart    {resume_from}" if resume_from else f"read_data       {data_file}"
    title = p.ensemble + " MD" if p.protocol == "single" else f"protocol {p.protocol}"
    lines = header(p, title + (" (resumed)" if resume_from else ""))
    lines += [f"log             {log_file} append", ""]
    lines += common_setup(p, data_cmd)
    lines += ["",
              f"timestep        {p.dt * FS:g}",
              "thermo_style    custom step time temp pe ke etotal press vol density pxx pyy pzz",
              "thermo_modify   lost error flush yes",
              f"thermo          {p.thermo}"]
    if p.protocol == "minimize":
        return "\n".join(lines + minimize_lines(p) + [""])
    lines += [""]
    if not resume_from:
        lines += [f"velocity        all create {p.temp} {p.seed} dist gaussian mom yes rot yes", ""]
    lines += ["variable        tnow equal temp",
              f"fix             runaway all halt {p.thermo} v_tnow > {halt:g} error soft",
              "",
              f"dump            traj all custom {p.dump} traj.lammpstrj id type element xu yu zu vx vy vz fx fy fz",
              f"dump_modify     traj element {' '.join(p.species)} sort id" + (" append yes" if resume_from else ""),
              f"restart         {p.restart} restart.a restart.b"]
    for st in plan:
        if resume_step is not None and st.end <= resume_step:
            continue
        lines += stage_lines(p, st, resume_step)
    lines += ["", "write_restart   final.restart", ""]
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
    protocol_sanity(p, atoms, rep)
    rep.check("dump interval", "warn" if p.steps // max(p.dump, 1) > 100_000 else "pass",
              f"{p.steps // max(p.dump, 1)} frames", suggest="more than 1e5 dump frames: raise -dump")


def protocol_sanity(p: MDParams, atoms, rep) -> None:
    full = all(atoms.pbc)
    if p.thermostat == "csvr":
        rep.check("thermostat", "pass", "Bussi (temp/csvr) + nve",
                  suggest=None)
        rep.data.setdefault("notes", []).append(
            "fix temp/csvr has a KOKKOS variant only from LAMMPS patch 2Sep2026 (stable 30Sep2026); "
            "older builds run it on the host, which is correct but slower")
    if p.ensemble == "npt" or p.protocol in ("equilibrate", "transport", "melt"):
        if p.baro == "tri" and not full:
            rep.check("barostat", "fail", "tri needs a fully periodic cell")
    if p.protocol in ("anneal", "quench"):
        ramp_ps = (0.4 if p.protocol == "anneal" else 0.8) * p.steps * p.dt * FS
        rate = abs(p.temp_end - p.temp) / max(ramp_ps, 1e-12)
        rep.check("ramp rate", "warn" if rate > 10 else "pass", f"{rate:.2f} K/ps",
                  suggest="ramp faster than 10 K/ps: glass transition and quenched structures depend on the "
                          "rate — run longer (more -steps) or report the rate with the result")
        rep.data["ramp_K_per_ps"] = rate
        if p.protocol == "quench" and p.temp_end >= p.temp:
            rep.check("quench", "fail", f"temp_end {p.temp_end:g} K is not below temp {p.temp:g} K")
        if abs(p.temp_end - p.temp) < 1e-9:
            rep.check("ramp", "fail", "-temp_end equals -temp: nothing to anneal / quench")
    if p.protocol == "melt":
        if not full or not is_orthogonal(atoms):
            rep.check("melt cell", "fail", "coexistence needs an orthogonal, fully periodic cell (interface ⟂ z)")
        else:
            lx, ly, lz = atoms.cell.lengths()
            rep.check("melt cell", "warn" if lz < 2 * max(lx, ly) else "pass",
                      f"{lx:.1f} × {ly:.1f} × {lz:.1f} Å, {len(atoms)} atoms",
                      suggest="coexistence boxes are usually elongated along the interface normal "
                              "(Lz ≈ 2–4 × Lx) with ~1e4 atoms: build a longer supercell along z")
        rep.check("melt temperature", "fail" if p.melt_temp <= p.temp else "pass",
                  f"liquid half at {p.melt_temp:g} K, coexistence started at {p.temp:g} K",
                  suggest="-melt_temp must be above the melting point (typically 1.5 × the guess) and below boiling")
    if p.protocol == "minimize" and p.ensemble != "nvt":
        rep.check("minimize", "pass", "-ensemble is ignored by minimize")


def cell_widths(atoms) -> np.ndarray:
    """Distance between opposite faces of the cell, per direction."""
    cell = np.array(atoms.cell)
    vol = abs(np.linalg.det(cell))
    return np.array([vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])
