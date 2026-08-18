#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
izo_formation.py — formation energy vs In ratio for In-Zn-O, from a MACE-evaluated xyz.

    E_f = [ E - (n_In/2)*E(In2O3) - n_Zn*E(ZnO) - dn_O*mu_O ] / N_atoms      eV/atom
    x   = n_In / (n_In + n_Zn)

Reading, logging, tables, colours and the plot style all come from mflow.py next door;
this file only holds what is specific to IZO.

    python tools/izo_formation.py -in izo_mace.xyz -eZnO zno_mace.xyz -eIn2O3 in2o3_mace.xyz
    python tools/izo_formation.py -in izo_mace.xyz -eZnO -8.12 -eIn2O3 -30.4 -muO -4.9 -xyz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import ase.io
from ase.formula import Formula

from mflow import COLORS, LOG, REF_CANDIDATES, fix_prefix, read_dataset, table   # noqa: E402

import matplotlib.pyplot as plt                                  # noqa: E402  (mflow sets the style)

ELEMENTS = ("In", "Zn", "O")
REFS = {"ZnO": Formula("ZnO").count(), "In2O3": Formula("In2O3").count()}
TOL = 1e-6                      # an atom count is an integer; anything above this is real


def die(message: str):
    LOG(f"[err]error[/err] {message}")
    sys.exit(1)


def sig3(value: float) -> str:
    return f"{value:.3g}"


# --- inputs ------------------------------------------------------------------ #

def energy_key(frames: list, prefix: str | None) -> str:
    """`-prefix mpa0_` if given, else the labelled energy in the file (bare `energy` last)."""
    if prefix:
        key = fix_prefix(prefix) + "energy"
        if key not in frames[0].info:
            die(f"no [key]{key}[/key] in the first structure — found {sorted(frames[0].info)}")
        return key
    found = sorted(k for k in frames[0].info if k == "energy" or k.endswith("_energy"))
    if not found:
        die("no energy label in the first structure — run `mflow.py calc` first, or pass -prefix")
    # dft_energy / REF_energy are what MACE was compared against, not what MACE produced,
    # so they only win when the file holds nothing else.
    reference_keys = {c + "energy" for c in REF_CANDIDATES}
    found.sort(key=lambda k: (k in reference_keys, k == "energy", k))
    if len(found) > 1:
        LOG(f"[warn]several energies present[/warn] {', '.join(found)} — using [key]{found[0]}[/key], "
            f"pass -prefix to choose")
    return found[0]


def reference(spec: str, name: str) -> float:
    """A number is eV per formula unit; anything else is an xyz we divide ourselves."""
    try:
        return float(spec)
    except ValueError:
        pass
    atoms = read_dataset(spec, index="0")[0]
    key = energy_key([atoms], None)
    have, stoich = atoms.symbols.formula.count(), REFS[name]
    extra = set(have) - set(stoich)
    units = {el: have.get(el, 0) / k for el, k in stoich.items()}
    n = min(units.values())
    if extra or n <= 0 or max(units.values()) - n > TOL:
        die(f"[key]{spec}[/key] is {atoms.symbols.formula}, not a clean {name} cell — "
            f"give the energy per formula unit as a number instead")
    value = float(atoms.info[key]) / n
    LOG(f"  [key]{name}[/key] {spec} · {atoms.symbols.formula} = {n:g} f.u. · "
        f"[hi]{value:.4f}[/hi] eV/f.u. (from {key})")
    return value


# --- the calculation --------------------------------------------------------- #

def formation(frames: list, ekey: str, e_zno: float, e_in2o3: float, mu_o: float | None):
    counts = [a.symbols.formula.count() for a in frames]
    stray = [i for i, c in enumerate(counts) if set(c) - set(ELEMENTS)]
    if stray:
        die(f"{len(stray)}/{len(frames)} structures contain elements outside In/Zn/O "
            f"(e.g. frame {stray[0]}: {frames[stray[0]].symbols.formula}) — "
            f"ZnO and In2O3 alone cannot balance them")

    n_in, n_zn, n_o = (np.array([c.get(e, 0) for c in counts], float) for e in ELEMENTS)
    cations = n_in + n_zn
    if np.any(cations == 0):
        die(f"{int(np.sum(cations == 0))} structures have no In or Zn — In ratio is undefined")

    dn_o = n_o - (1.5 * n_in + n_zn)
    off = np.abs(dn_o) > TOL
    if off.any() and mu_o is None:
        die(f"{off.sum()}/{len(frames)} structures are not stoichiometric "
            f"(excess O from {dn_o[off].min():+g} to {dn_o[off].max():+g}) — "
            f"pass [key]-muO[/key] (eV per O atom) so the extra/missing oxygen can be referenced")
    if off.any():
        LOG(f"[warn]{off.sum()}/{len(frames)} structures off stoichiometry[/warn] · "
            f"excess O {dn_o[off].min():+g} … {dn_o[off].max():+g} · referenced at muO = {mu_o:g} eV")

    energy = np.array([float(a.info[ekey]) for a in frames])
    n_atoms = np.array([len(a) for a in frames], float)
    e_f = (energy - n_in / 2 * e_in2o3 - n_zn * e_zno - dn_o * (mu_o or 0.0)) / n_atoms
    return n_in / cations, e_f, energy, n_atoms


def per_ratio(x: np.ndarray, e_f: np.ndarray):
    """Group by In ratio; return the ratios, their lowest/mean/spread and the winning frame."""
    ratios, inv = np.unique(np.round(x, 6), return_inverse=True)
    counts = np.bincount(inv)
    low = np.full(len(ratios), np.inf)
    high = np.full(len(ratios), -np.inf)
    np.minimum.at(low, inv, e_f)
    np.maximum.at(high, inv, e_f)
    mean = np.bincount(inv, weights=e_f) / counts
    order = np.lexsort((e_f, inv))                    # per group, lowest energy first
    best = order[np.searchsorted(inv[order], np.arange(len(ratios)))]
    return ratios, counts, low, mean, high - low, best


# --- output ------------------------------------------------------------------ #

def report(frames, ratios, counts, low, mean, spread, best):
    LOG(table("formation energy by In ratio",
              ["x(In)", "structures", "lowest formula", "E_f min", "mean", "spread", "frame"],
              [[f"{r:.3f}", c, str(frames[b].symbols.formula), sig3(lo), sig3(m), sig3(s), b]
               for r, c, lo, m, s, b in zip(ratios, counts, low, mean, spread, best)],
              styles=["key", None, None, "ok"]))
    i = int(np.argmin(low))
    LOG(f"most stable · [key]x = {ratios[i]:.3f}[/key] · frame [hi]{best[i]}[/hi] "
        f"{frames[best[i]].symbols.formula} · [ok]{sig3(low[i])}[/ok] eV/atom")


def write_csv(path: Path, frames, x, e_f, energy, n_atoms):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "formula", "n_atoms", "x_In", "E_total_eV", "E_f_eV_per_atom"])
        for i, atoms in enumerate(frames):
            writer.writerow([i, str(atoms.symbols.formula), int(n_atoms[i]),
                             f"{x[i]:.6f}", f"{energy[i]:.6f}", f"{e_f[i]:.6f}"])


def make_plot(path: Path, x, e_f, ratios, low, unit: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    size = 70 if len(x) < 200 else (30 if len(x) < 1000 else (12 if len(x) < 20_000 else 5))
    ax.axhline(0, ls="--", lw=1.5, color="0.55", zorder=0)
    ax.scatter(x, e_f, s=size, alpha=0.7, color=COLORS[0], edgecolors="none",
               label=f"{len(x)} structures", zorder=2)
    ax.plot(ratios, low, "-o", color=COLORS[1], ms=8, mew=2, zorder=3,
            label="lowest at each ratio")
    ax.set_xlabel("In / (In + Zn)")
    ax.set_ylabel(f"formation energy ({unit})")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Formation energy vs In ratio for In-Zn-O structures already evaluated by MACE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="the two references take either a number (eV per formula unit) or an xyz to read")
    p.add_argument("-in", dest="infile", required=True, help="xyz with MACE energies")
    p.add_argument("-eZnO", required=True, help="E(ZnO) in eV per formula unit, or an xyz")
    p.add_argument("-eIn2O3", required=True, help="E(In2O3) in eV per formula unit, or an xyz")
    p.add_argument("-muO", type=float, default=None,
                   help="oxygen reference in eV/atom, needed if any structure is off stoichiometry")
    p.add_argument("-prefix", default=None, help="energy label to use, e.g. mpa0_ (default: auto)")
    p.add_argument("-out", default=None, help="output stem (default: alongside the input)")
    p.add_argument("-xyz", action="store_true", help="also write an xyz tagged with izo_x_In / izo_Ef")
    args = p.parse_args(argv)

    stem = Path(args.out) if args.out else Path(args.infile).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    LOG.rule("IZO formation energy")
    frames = read_dataset(args.infile)
    ekey = energy_key(frames, args.prefix)
    LOG(f"energy label [key]{ekey}[/key] · references")
    e_zno = reference(args.eZnO, "ZnO")
    e_in2o3 = reference(args.eIn2O3, "In2O3")

    x, e_f, energy, n_atoms = formation(frames, ekey, e_zno, e_in2o3, args.muO)
    grouped = per_ratio(x, e_f)
    report(frames, *grouped)

    csv_path, png_path = Path(f"{stem}_izo_ef.csv"), Path(f"{stem}_izo_ef.png")
    write_csv(csv_path, frames, x, e_f, energy, n_atoms)
    make_plot(png_path, x, e_f, grouped[0], grouped[2], "eV/atom")
    written = [csv_path, png_path]

    if args.xyz:
        for atoms, xi, ef in zip(frames, x, e_f):
            atoms.info["izo_x_In"], atoms.info["izo_Ef_atom"] = float(xi), float(ef)
        out = Path(f"{stem}_izo_ef.xyz")
        ase.io.write(out, frames, format="extxyz")
        written.append(out)

    LOG("wrote " + " · ".join(f"[key]{w}[/key]" for w in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
