# -*- coding: utf-8 -*-
"""
Trajectory observables from a LAMMPS custom dump (id type element xu yu zu vx vy vz fx fy fz).

Positions in the dump are unwrapped (xu yu zu) — used as is for MSD, wrapped back into the
cell for distances and RDF. Everything is plain numpy + ASE neighbour lists, no extra deps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ase.io


def read_dump(path: str | Path, stride: int = 1, species: list | None = None) -> list:
    frames = ase.io.read(path, index=f"::{max(stride, 1)}", format="lammps-dump-text",
                         **({"specorder": species} if species else {}))
    return frames if isinstance(frames, list) else [frames]


def wrapped(atoms):
    a = atoms.copy()
    a.wrap()
    return a


def min_distances(frames: list, cutoff: float = 2.5) -> np.ndarray:
    from ase.neighborlist import neighbor_list
    out = []
    for atoms in frames:
        d = neighbor_list("d", wrapped(atoms), cutoff)
        out.append(float(d.min()) if d.size else cutoff)
    return np.array(out)


def max_rdf_range(atoms) -> float:
    """Half the thinnest periodic width: beyond it the minimum-image RDF double counts."""
    from ..lmp.inputs import cell_widths
    return 0.5 * float(min(cell_widths(atoms)))


def rdf(frames: list, rmax: float | None = None, nbins: int = 200) -> dict:
    """Partial g(r) per element pair, averaged over frames. Returns {"r": .., "A-B": g, ...}."""
    from ase.neighborlist import neighbor_list
    rmax = min(rmax or 8.0, max_rdf_range(frames[0]) - 1e-6)
    edges = np.linspace(0, rmax, nbins + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    symbols = sorted(set(frames[0].get_chemical_symbols()))
    hist = {f"{a}-{b}": np.zeros(nbins) for i, a in enumerate(symbols) for b in symbols[i:]}
    norm = {k: 0.0 for k in hist}
    for atoms in frames:
        atoms = wrapped(atoms)
        sym = np.array(atoms.get_chemical_symbols())
        vol = atoms.get_volume()
        i, j, d = neighbor_list("ijd", atoms, rmax)
        for key in hist:
            a, b = key.split("-")
            m = (sym[i] == a) & (sym[j] == b)
            hist[key] += np.histogram(d[m], bins=edges)[0]
            na, nb = (sym == a).sum(), (sym == b).sum()
            norm[key] += na * nb / vol            # ordered pairs a->b, so no factor 2 needed
    out = {"r": r}
    for key in hist:
        out[key] = hist[key] / (shell * norm[key]) if norm[key] else hist[key]
    return out


def msd(frames: list, dt_ps: float, origins: int = 10) -> dict:
    """
    Mean squared displacement per element (Å²) from unwrapped positions, averaged over
    `origins` evenly spaced time origins. Lag times in ps.
    """
    pos = np.array([a.get_positions() for a in frames])           # (T, N, 3) unwrapped
    sym = np.array(frames[0].get_chemical_symbols())
    T = len(frames)
    max_lag = T // 2
    if max_lag < 2:
        return {}
    starts = np.unique(np.linspace(0, T - max_lag - 1, min(origins, T - max_lag)).astype(int))
    lags = np.arange(max_lag)
    out = {"t_ps": lags * dt_ps}
    for el in sorted(set(sym)):
        m = sym == el
        acc = np.zeros(max_lag)
        for s in starts:
            d = pos[s:s + max_lag, m] - pos[s, m]
            acc += (d ** 2).sum(-1).mean(-1)
        out[el] = acc / len(starts)
    return out


def diffusion(msd_curve: dict, fit_from: float = 0.2) -> dict:
    """D per element (cm²/s) from the slope of the later part of the MSD: D = slope / 6."""
    t = msd_curve.get("t_ps")
    if t is None or len(t) < 5:
        return {}
    k = int(len(t) * fit_from)
    out = {}
    for el, y in msd_curve.items():
        if el == "t_ps":
            continue
        slope = np.polyfit(t[k:], y[k:], 1)[0]                     # Å²/ps
        out[el] = float(slope / 6 * 1e-4)                           # Å²/ps -> cm²/s
    return out


def rdf_difference(a: dict, b: dict) -> dict:
    """Mean absolute g(r) difference per common pair, on a's r grid."""
    out = {}
    for key in a:
        if key == "r" or key not in b:
            continue
        gb = np.interp(a["r"], b["r"], b[key])
        out[key] = float(_trapz(np.abs(a[key] - gb), a["r"]) / (a["r"][-1] - a["r"][0]))
    return out


_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy < 2.0 has only trapz
