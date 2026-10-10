# -*- coding: utf-8 -*-
"""
mflow validate — layer-1 checks: static properties a potential must get right before any MD.

    dimer    E(r) and F(r) of every element pair, isolated atoms subtracted, on MLIP Arena's grid
             (0.9 r_cov → 3.1 r_vdW, 0.01 Å) with its shape metrics (force flips, energy jumps,
             tortuosity, Spearman of the repulsive wall, conservation |F + dE/dr|). A physical curve
             has one minimum, a monotonic repulsive wall, forces equal to -dE/dr and E -> 0 beyond
             the cutoff. Holes in the short-range wall are what makes MD collapse atoms.
    eos      E(V) under uniform scaling, Birch–Murnaghan fit: V0, B0, B0'. The minimum must lie
             inside the sampled range.
    phonon   finite displacements with phonopy (optional dependency): imaginary modes on a q-mesh,
             highest frequency, band structure + DOS. Positions are relaxed first (cell fixed,
             symmetry kept) — phonons of an unrelaxed structure are meaningless.
    defect   vacancy formation energy per element (elemental solids; for compounds the raw
             E(N-1) - E(N) is reported, a formation energy needs chemical potentials) and surface
             energies for the Miller indices you ask for (indices refer to the cell you give:
             pass the conventional cell for cubic crystals).

-ref "B0=140,V0=11.8,Ev_Cu=1.28,gamma_111=1.3" compares with your DFT numbers (thresholds.yaml
section validate). Everything is written to <stem>_<kind>.json (+ png).
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms, units
from ase.data import atomic_numbers, covalent_radii

from . import __version__
from .log import LOG, elapsed, num, table
from .report import Report
from .style import COLORS, LINESTYLES, plt
from .thresholds import section

EV_A2_TO_J_M2 = 16.02176634


# ============================================================================ #
#  helpers
# ============================================================================ #

def calculator(model: str, head: str | None, dtype: str = "float64", device: str = "auto"):
    """MACE calculator; `emt` gives ASE's EMT (for trying the commands without a model)."""
    if model.lower() == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    from .model import build_calculator, pick_device
    return build_calculator(model, pick_device(device), dtype, head)


def parse_ref(text: str | None) -> dict:
    out = {}
    for item in (text or "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = float(v)
    return out


def compare_ref(rep: Report, name: str, value: float, ref: float | None, unit: str):
    if ref is None:
        return
    th = section("validate")
    rel = abs(value - ref) / max(abs(ref), 1e-12)
    status = "pass" if rel < th["ref_pass"] else ("warn" if rel < th["ref_warn"] else "fail")
    rep.check(f"{name} vs reference", status, f"{value:.4g} vs {ref:.4g} {unit} ({rel * 100:.1f} %)",
              value=rel, suggest=f"{name} is off the reference: add training data around this deformation "
                                 f"(strained / defected cells) and check that the DFT settings match")


def relax_positions(atoms: Atoms, calc, fmax: float, steps: int = 500, symmetric: bool = True) -> Atoms:
    from ase.optimize import BFGS
    a = atoms.copy()
    a.calc = calc
    if symmetric:
        try:
            from ase.constraints import FixSymmetry
            a.set_constraint(FixSymmetry(a))
        except Exception:                           # spglib missing: relax without symmetry
            pass
    BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    a.set_constraint()
    return a


def input_state(atoms: Atoms, calc, rep: Report):
    """Is the input a relaxed structure? (forces, and stress for periodic cells)"""
    a = atoms.copy()
    a.calc = calc
    fmax = float(np.abs(a.get_forces()).max())
    msg = f"max |F| {fmax:.3f} eV/Å"
    smax = None
    if all(a.pbc):
        smax = float(np.abs(a.get_stress()).max() / units.GPa)
        msg += f" · max |σ| {smax:.2f} GPa"
    bad = fmax > 0.05 or (smax is not None and smax > 0.5)
    rep.check("input relaxed", "warn" if bad else "pass", msg,
              suggest="the input is not at a minimum of this model: relax it first "
                      "(mflow relax -in ... -model ...), otherwise V0 / phonons / defect energies are biased")
    return fmax, smax


# ============================================================================ #
#  dimer
# ============================================================================ #

def isolated(calc, element: str) -> float:
    a = Atoms(element, positions=[[0, 0, 0]], cell=[30, 30, 30], pbc=False)
    a.calc = calc
    return float(a.get_potential_energy())


def dimer_curve(calc, a: str, b: str, rs: np.ndarray, e_iso: dict) -> dict:
    E, F = [], []
    for r in rs:
        d = Atoms(a + b, positions=[[0, 0, 0], [r, 0, 0]], cell=[30, 30, 30], pbc=False)
        d.calc = calc
        E.append(float(d.get_potential_energy()) - e_iso[a] - e_iso[b])
        F.append(float(d.get_forces()[1, 0]))       # on atom 2 along +x: > 0 repulsive
    return {"r": rs, "E": np.array(E), "F": np.array(F)}


def dimer_metrics(r: np.ndarray, E: np.ndarray, F: np.ndarray) -> dict:
    """
    Shape metrics as MLIP Arena computes them (mlip_arena/flows/diatomics.py), plus two of ours:
      conservation_deviation   mean |F + dE/dr|                                      ideal 0
      force_flips              sign changes of F, |F| < 0.01 eV/Å dropped             ideal 1
      energy_diff_flips        sign changes of ΔE, |ΔE| < 1 meV dropped               ideal 1
      energy_jump              Σ (|ΔE_i| + |ΔE_i+1|) at each ΔE sign change (eV)     ideal ~0
      tortuosity               Σ|ΔE| / (|E(r_max) - E_min| + (E(r_min) - E_min))    ideal 1
      spearman_repulsion       ρ(r, E) for r ≤ r(E_min)                               ideal -1
      spearman_attraction      ρ(r, E) for r > r(E_min)                               ideal +1
      spearman_force_descending ρ(r, F) for r ≤ r(F_min)                              ideal -1
      repulsion_violations     (ours) steps where E rises with r below r(E_min): holes in the wall
      E_short / E_tail         (ours) E at the first / last distance, isolated atoms subtracted
    """
    from .uq import spearman
    o = np.argsort(r)
    r, E, F = np.asarray(r)[o], np.asarray(E)[o], np.asarray(F)[o]
    i_e, i_f = int(np.argmin(E)), int(np.argmin(F))
    dE = np.diff(E)

    def flips(x):
        s = np.sign(x)
        return int(np.count_nonzero(np.diff(s))) if len(s) > 1 else 0

    Fz = F[np.abs(F) >= 0.01]
    dEz = dE[np.abs(dE) >= 1e-3]
    at = np.where(np.diff(np.sign(dEz)) != 0)[0]
    span = abs(E[-1] - E[i_e]) + (E[0] - E[i_e])
    rho = lambda x, y: float(spearman(x, y)) if len(x) > 2 else None        # None: segment too short
    return {"r_eq": float(r[i_e]), "well_depth": float(E[-1] - E[i_e]), "E_short": float(E[0]), "E_tail": float(E[-1]),
            "conservation_deviation": float(np.mean(np.abs(F + np.gradient(E, r)))),
            "force_flips": flips(Fz), "energy_diff_flips": flips(dEz),
            "energy_jump": float(sum(abs(dEz[i]) + abs(dEz[i + 1]) for i in at)),
            "tortuosity": float(np.abs(dE).sum() / span) if span > 0 else float("inf"),
            "spearman_repulsion": rho(r[:i_e + 1], E[:i_e + 1]),
            "spearman_attraction": rho(r[i_e + 1:], E[i_e + 1:]),
            "spearman_force_descending": rho(r[:i_f + 1], F[:i_f + 1]),
            "repulsion_violations": int(np.sum(dE[:i_e] > 0))}


def dimer_grid(a: str, b: str, rmin: float, rmax: float | None, dr: float) -> np.ndarray:
    """MLIP Arena: 0.9 × covalent radius → 3.1 × van der Waals radius (Alvarez), 0.01 Å apart."""
    from ase.data.vdw_alvarez import vdw_radii
    za, zb = atomic_numbers[a], atomic_numbers[b]
    lo = rmin * 0.5 * (covalent_radii[za] + covalent_radii[zb])
    if rmax is None:
        vdw = 0.5 * (vdw_radii[za] + vdw_radii[zb])
        rmax = 3.1 * vdw if np.isfinite(vdw) else 6.0
    return np.linspace(lo, rmax, max(int((rmax - lo) / dr), 20))


def run_dimer(calc, elements: list, rmin: float, rmax: float | None, dr: float, rep: Report) -> dict:
    e_iso = {el: isolated(calc, el) for el in elements}
    th = section("validate")
    out, rows = {}, []
    for a, b in itertools.combinations_with_replacement(elements, 2):
        rs = dimer_grid(a, b, rmin, rmax, dr)
        curve = dimer_curve(calc, a, b, rs, e_iso)
        m = dimer_metrics(curve["r"], curve["E"], curve["F"])
        out[f"{a}-{b}"] = {**m, "r": rs.tolist(), "E": curve["E"].tolist(), "F": curve["F"].tolist()}
        wall = []
        if m["E_short"] <= m["E_tail"]:
            wall.append(f"no repulsive wall (E({rs[0]:.2f} Å) ≤ E(∞))")
        if m["repulsion_violations"]:
            wall.append(f"{m['repulsion_violations']} hole(s) in the wall")
        rep.check(f"dimer {a}-{b}: wall", "fail" if wall else "pass",
                  "; ".join(wall) or f"repulsive down to {rs[0]:.2f} Å (ρ {num(m['spearman_repulsion'], 3)})",
                  suggest="the short-range wall is broken: MD can collapse atoms (closest-pair gate). Add "
                          "compressed / close-contact configurations to training, or use a model with a "
                          "repulsive prior (ZBL)")
        shape = []
        if m["force_flips"] != 1:
            shape.append(f"{m['force_flips']} force sign changes")
        if m["energy_diff_flips"] > 1:
            shape.append(f"{m['energy_diff_flips']} extrema")
        if m["energy_jump"] > th["dimer_jump_eV"]:
            shape.append(f"energy jump {m['energy_jump']:.3f} eV")
        if m["conservation_deviation"] > th["dimer_conservation_eV_A"]:
            shape.append(f"|F + dE/dr| {m['conservation_deviation']:.3f} eV/Å")
        if abs(m["E_tail"]) > th["dimer_tail_eV"]:
            shape.append(f"E({rs[-1]:.1f} Å) = {m['E_tail']:.3f} eV vs isolated atoms")
        rep.check(f"dimer {a}-{b}: shape", "warn" if shape else "pass",
                  "; ".join(shape) or f"r_eq {m['r_eq']:.2f} Å · depth {m['well_depth']:.2f} eV · smooth",
                  suggest="dimer curve not smooth (extra extrema, jumps or forces that are not -dE/dr): expect "
                          "poor energy conservation in NVE where such distances occur")
        rows.append((f"{a}-{b}", f"{m['r_eq']:.2f}", f"{m['well_depth']:.3f}", m["force_flips"],
                     m["energy_diff_flips"], f"{m['tortuosity']:.2f}", num(m["spearman_repulsion"], 3),
                     f"{m['conservation_deviation']:.3f}"))
    LOG(table("dimer curves (MLIP Arena metrics)",
              ["pair", "r_eq (Å)", "depth (eV)", "F flips", "ΔE flips", "tortuosity", "ρ repulsion", "|F+dE/dr|"],
              rows, styles=["key"]))
    return {"isolated_eV": e_iso, "pairs": out}


def plot_dimer(data: dict, outfile: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (pair, d) in enumerate(data["pairs"].items()):
        c = COLORS[i % len(COLORS)]
        ax1.plot(d["r"], d["E"], color=c, label=pair)
        ax2.plot(d["r"], d["F"], color=c, label=pair)
    depth = max([d["well_depth"] for d in data["pairs"].values()] + [1.0])
    ax1.set_ylim(-1.5 * depth, 3 * depth)
    ax2.set_ylim(-5 * depth, 10 * depth)
    for ax in (ax1, ax2):
        ax.axhline(0, color="black", lw=0.8, ls=LINESTYLES[1])
        ax.tick_params(which="both", direction="in")
        ax.legend()
    ax1.set(xlabel=r"r ($\mathrm{\AA}$)", ylabel="E - E(isolated) (eV)")
    ax2.set(xlabel=r"r ($\mathrm{\AA}$)", ylabel=r"F$_r$ (eV/$\mathrm{\AA}$), > 0 repulsive")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


# ============================================================================ #
#  eos
# ============================================================================ #

def run_eos(calc, atoms: Atoms, strain: float, npts: int, relax: bool, rep: Report, ref: dict) -> dict:
    from ase.eos import EquationOfState
    scales = np.linspace(1 - strain, 1 + strain, npts)          # linear strain
    V, E = [], []
    for s in scales:
        a = atoms.copy()
        a.set_cell(np.array(atoms.cell) * s, scale_atoms=True)
        if relax:
            a = relax_positions(a, calc, fmax=0.01, symmetric=True)
        a.calc = calc
        V.append(a.get_volume() / len(a))
        E.append(a.get_potential_energy() / len(a))
    V, E = np.array(V), np.array(E)
    eos = EquationOfState(V, E, eos="birchmurnaghan")
    v0, e0, B = eos.fit()
    bp = float(eos.eos_parameters[2]) if getattr(eos, "eos_parameters", None) is not None else None
    B_GPa = float(B / units.GPa)
    fit = np.array([eos.func(v, *eos.eos_parameters) for v in V]) if bp is not None else None
    rmse = float(np.sqrt(np.mean((fit - E) ** 2)) * 1000) if fit is not None else None
    inside = V.min() < v0 < V.max()
    rep.check("eos minimum inside range", "pass" if inside else "fail",
              f"V0 {v0:.3f} Å³/atom, sampled {V.min():.3f}–{V.max():.3f}",
              suggest="the E(V) minimum is outside the sampled volumes: relax the cell first or widen -strain")
    i_min = int(np.argmin(E))
    rep.check("eos single minimum", "pass" if 0 < i_min < len(E) - 1 and np.all(np.diff(np.sign(np.diff(E))) >= 0)
              else "warn", "E(V) convex" if np.all(np.diff(np.sign(np.diff(E))) >= 0) else "E(V) not convex",
              suggest="E(V) has more than one minimum or kinks: possible phase change or a defect of the "
                      "potential under strain — add strained cells to training")
    if rmse is not None:
        rep.check("eos fit", "pass" if rmse < 1.0 else "warn", f"RMSE {rmse:.3f} meV/atom",
                  suggest="Birch–Murnaghan does not describe E(V) well: reduce -strain or look for kinks")
    compare_ref(rep, "V0", float(v0), ref.get("V0"), "Å³/atom")
    compare_ref(rep, "B0", B_GPa, ref.get("B0"), "GPa")
    LOG(table("equation of state (Birch–Murnaghan)", ["", "value", "unit"],
              [("V0", f"{v0:.4f}", "Å³/atom"), ("E0", f"{e0:.5f}", "eV/atom"), ("B0", f"{B_GPa:.1f}", "GPa"),
               ("B0'", f"{bp:.2f}" if bp is not None else "—", ""), ("fit RMSE", num(rmse, 3) if rmse else "—", "meV/atom")],
              styles=["key", None, "dim"]))
    return {"V": V.tolist(), "E": E.tolist(), "V0_A3_atom": float(v0), "E0_eV_atom": float(e0), "B0_GPa": B_GPa,
            "B0_prime": bp, "fit_rmse_meV_atom": rmse, "relaxed_positions": relax, "strain": strain,
            "_params": list(eos.eos_parameters) if bp is not None else None}


def plot_eos(data: dict, outfile: str):
    from ase.eos import birchmurnaghan
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(data["V"], data["E"], "o", color=COLORS[0], label="model")
    if data.get("_params"):
        v = np.linspace(min(data["V"]), max(data["V"]), 200)
        ax.plot(v, birchmurnaghan(v, *data["_params"]), color=COLORS[1], label="Birch–Murnaghan")
    ax.axvline(data["V0_A3_atom"], color="black", lw=0.8, ls=LINESTYLES[1])
    ax.set(xlabel=r"Volume ($\mathrm{\AA}^3$/atom)", ylabel="Energy (eV/atom)",
           title=f"V0 {data['V0_A3_atom']:.3f} Å³/atom · B0 {data['B0_GPa']:.0f} GPa")
    ax.tick_params(which="both", direction="in")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


# ============================================================================ #
#  phonon
# ============================================================================ #

def run_phonon(calc, atoms: Atoms, supercell: list, mesh: int, distance: float, rep: Report) -> dict:
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        rep.check("phonopy", "fail", "phonopy is not installed", suggest="pip install phonopy seekpath")
        return {}
    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=np.array(atoms.cell),
                        scaled_positions=atoms.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=np.diag(supercell), primitive_matrix="auto")
    ph.generate_displacements(distance=distance)
    cells = ph.supercells_with_displacements
    LOG(f"phonopy: supercell {'×'.join(map(str, supercell))} ({len(cells[0])} atoms) · {len(cells)} displacement(s)")
    forces = []
    for sc in cells:
        a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        f = a.get_forces()
        forces.append(f - f.mean(0))                    # remove the residual drift
    ph.forces = forces
    ph.produce_force_constants()
    ph.run_mesh([mesh] * 3, is_gamma_center=True, with_eigenvectors=False)
    freqs = np.asarray(ph.mesh.frequencies)
    tol = section("validate")["imaginary_THz"]
    imag = freqs < -tol
    n_imag_q = int(np.any(imag, axis=1).sum())
    status = "pass" if not imag.any() else ("warn" if freqs.min() > -5 * tol else "fail")
    rep.check("imaginary modes", status,
              f"{n_imag_q} of {len(freqs)} q-points · lowest {freqs.min():.3f} THz (tolerance -{tol:g})",
              value=float(freqs.min()),
              suggest="imaginary phonons: the structure is not a minimum of this potential (relax tighter, larger "
                      "supercell) or the potential is wrong for this phase — compare with DFT phonons and add "
                      "displaced / strained cells to training")
    out = {"supercell": supercell, "mesh": mesh, "displacement_A": distance, "n_displacements": len(cells),
           "min_THz": float(freqs.min()), "max_THz": float(freqs.max()), "n_imaginary_qpoints": n_imag_q}
    ph.run_total_dos()
    out["_dos"] = (np.asarray(ph.total_dos.frequency_points).tolist(), np.asarray(ph.total_dos.dos).tolist())
    try:
        ph.auto_band_structure(npoints=51)
        bs = ph.band_structure
        out["_bands"] = ([np.asarray(d).tolist() for d in bs.distances],
                         [np.asarray(f).tolist() for f in bs.frequencies],
                         tick_labels(list(bs.labels or []), list(bs.path_connections or [])))
    except Exception as exc:                            # needs seekpath
        LOG(f"[dim]band structure skipped ({type(exc).__name__}: {exc}); pip install seekpath[/dim]")
    LOG(table("phonons", ["", "value"], [("lowest (THz)", f"{freqs.min():.3f}"), ("highest (THz)", f"{freqs.max():.2f}"),
                                         ("q-points with imaginary modes", f"{n_imag_q} / {len(freqs)}")],
              styles=["key"]))
    return out


def tick_labels(labels: list, connections: list) -> list:
    """One label per segment boundary; a break in the path (connection False) shows as 'X|U'."""
    if not labels:
        return []
    out, k = [labels[0]], 1
    for i, conn in enumerate(connections):
        last = i == len(connections) - 1
        if conn or last or k + 1 >= len(labels):
            out.append(labels[k] if k < len(labels) else "")
            k += 1
        else:
            out.append(f"{labels[k]}|{labels[k + 1]}")
            k += 2
    return out


def plot_phonon(data: dict, outfile: str):
    has_bands = "_bands" in data
    fig, axes = plt.subplots(1, 2 if has_bands else 1, figsize=(11 if has_bands else 6, 4.8), squeeze=False, sharey=True,
                             gridspec_kw={"width_ratios": [3, 1]} if has_bands else None)
    axes = axes[0]
    if has_bands:
        ax = axes[0]
        dist, freq, labels = data["_bands"]
        ticks = [dist[0][0]] + [d[-1] for d in dist]
        for d, f in zip(dist, freq):
            ax.plot(d, f, color=COLORS[0], lw=1.2)
        for x in ticks:
            ax.axvline(x, color="grey", lw=0.6)
        if len(labels) == len(ticks):
            ax.set_xticks(ticks, labels)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set(ylabel="Frequency (THz)")
    ax = axes[-1]
    x, y = data["_dos"]
    ax.plot(y, x, color=COLORS[1])
    ax.set(xlabel="DOS", ylabel="" if has_bands else "Frequency (THz)")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8, ls=LINESTYLES[1])
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


# ============================================================================ #
#  defects and surfaces
# ============================================================================ #

def repeat_for(atoms: Atoms, min_width: float) -> tuple:
    from .lmp.inputs import cell_widths
    return tuple(int(np.ceil(min_width / w)) for w in cell_widths(atoms))


def run_defect(calc, atoms: Atoms, min_width: float, surfaces: list, layers: int, rep: Report, ref: dict) -> dict:
    out = {"vacancy": {}, "surface": {}}
    a = atoms.copy()
    a.calc = calc
    e_bulk_atom = a.get_potential_energy() / len(a)
    rep_n = repeat_for(atoms, min_width)
    sc = atoms.repeat(rep_n)
    sc.calc = calc
    n = len(sc)
    e_bulk = sc.get_potential_energy()
    elemental = len(set(atoms.get_chemical_symbols())) == 1
    rows = []
    for el in sorted(set(atoms.get_chemical_symbols()), key=lambda s: atomic_numbers[s]):
        d = sc.copy()
        del d[[i for i, s in enumerate(d.get_chemical_symbols()) if s == el][0]]
        d = relax_positions(d, calc, fmax=0.01, steps=300, symmetric=False)
        d.calc = calc
        e_def = d.get_potential_energy()
        raw = e_def - e_bulk
        ef = e_def - (n - 1) / n * e_bulk if elemental else None
        out["vacancy"][el] = {"supercell": list(rep_n), "natoms": n, "E_def_minus_bulk_eV": float(raw),
                              "E_formation_eV": None if ef is None else float(ef)}
        rows.append((f"vacancy {el}", f"{ef:.3f}" if ef is not None else f"{raw:.3f} (raw)", "eV", f"{n} atoms"))
        if ef is not None:
            rep.check(f"vacancy {el}", "pass" if ef > 0 else "fail", f"E_f {ef:.3f} eV",
                      suggest="negative vacancy formation energy: the potential is unstable for defected cells — "
                              "add vacancy configurations to training")
            compare_ref(rep, f"Ev_{el}", ef, ref.get(f"Ev_{el}"), "eV")
    for hkl in surfaces:
        from ase.build import surface
        slab = surface(atoms, hkl, layers, vacuum=10.0, periodic=True)
        slab = relax_positions(slab, calc, fmax=0.02, steps=300, symmetric=False)
        slab.calc = calc
        area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
        gamma = (slab.get_potential_energy() - len(slab) * e_bulk_atom) / (2 * area) * EV_A2_TO_J_M2
        key = "".join(map(str, hkl))
        out["surface"][key] = {"layers": layers, "natoms": len(slab), "gamma_J_m2": float(gamma)}
        rows.append((f"surface ({key})", f"{gamma:.3f}", "J/m²", f"{len(slab)} atoms"))
        rep.check(f"surface ({key})", "pass" if gamma > 0 else "fail", f"γ {gamma:.3f} J/m²",
                  suggest="negative surface energy: the slab is more stable than bulk for this potential — "
                          "add surface configurations to training")
        compare_ref(rep, f"gamma_{key}", gamma, ref.get(f"gamma_{key}"), "J/m²")
    if not elemental:
        LOG("[dim]compound: vacancy formation energies need chemical potentials, the raw E(N-1) - E(N) is "
            "reported; surface energies assume a stoichiometric slab[/dim]")
    LOG(table("defects", ["", "value", "unit", "cell"], rows, styles=["key", None, "dim", "dim"]))
    return out


# ============================================================================ #
#  command
# ============================================================================ #

def cmd_validate(args) -> int:
    t0 = time.perf_counter()
    kind = args.kind
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    LOG.rule(f"MFlow {__version__} · validate {kind}")
    calc = calculator(args.model, args.head, args.dtype)
    ref = parse_ref(args.ref)
    rep = Report(f"validate-{kind}", model=args.model, head=args.head)
    if kind == "dimer":
        if args.elements:
            elements = [e.strip() for e in args.elements.split(",") if e.strip()]
        elif args.input:
            elements = sorted({s for a in ase.io.read(args.input, ":") for s in a.get_chemical_symbols()},
                              key=lambda s: atomic_numbers[s])
        else:
            LOG("[err]dimer needs -elements Cu,Al or -in structure.xyz[/err]")
            return 2
        stem = "dimer_" + "".join(elements)
        LOG.kv({"model": args.model, "elements": " ".join(elements),
                "r": f"{args.rmin:g}·r_cov → {f'{args.rmax:g} Å' if args.rmax else '3.1·r_vdW'}, every {args.dr:g} Å"})
        data = run_dimer(calc, elements, args.rmin, args.rmax, args.dr, rep)
        if not args.no_plot:
            plot_dimer(data, str(outdir / f"{stem}.png"))
    else:
        if not args.input:
            LOG(f"[err]{kind} needs -in structure.xyz[/err]")
            return 2
        atoms = ase.io.read(args.input, args.index)
        stem = f"{Path(args.input).stem}_{kind}"
        LOG.kv({"model": args.model, "structure": f"{args.input} [{args.index}]", "formula": atoms.get_chemical_formula(),
                "atoms": len(atoms)})
        if kind in ("eos", "phonon", "defect") and not all(atoms.pbc):
            rep.check("periodic", "fail", f"{kind} needs a fully periodic crystal")
            rep.show()
            rep.write(outdir / f"{stem}.json")
            return 2
        input_state(atoms, calc, rep)
        if kind == "eos":
            data = run_eos(calc, atoms, args.strain, args.npts, args.relax, rep, ref)
            if not args.no_plot:
                plot_eos(data, str(outdir / f"{stem}.png"))
        elif kind == "phonon":
            sc = [int(x) for x in args.supercell.split(",")] if args.supercell else list(repeat_for(atoms, 10.0))
            sc = sc * 3 if len(sc) == 1 else sc
            work = relax_positions(atoms, calc, fmax=1e-3) if args.relax else atoms
            data = run_phonon(calc, work, sc, args.mesh, args.displacement, rep)
            if data and not args.no_plot:
                plot_phonon(data, str(outdir / f"{stem}.png"))
        else:
            surfaces = [tuple(int(c) for c in s) for s in (args.surface.split(",") if args.surface else [])]
            data = run_defect(calc, atoms, args.min_width, surfaces, args.layers, rep, ref)
    rep.data[kind] = {k: v for k, v in data.items() if not k.startswith("_")}
    rep.data["reference"] = ref
    LOG(f"total [hi]{elapsed(time.perf_counter() - t0)}[/hi]")
    LOG.rule("verdict")
    rep.show()
    rep.write(outdir / f"{stem}.json")
    return 0 if not rep.failed else 2
