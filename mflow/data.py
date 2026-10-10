# -*- coding: utf-8 -*-
"""Reading datasets, label prefixes, compositions and grouping."""

from __future__ import annotations

import numpy as np
import ase.io
from ase import Atoms

from .log import LOG

REF_CANDIDATES = ["", "REF_", "ref_", "dft_", "DFT_", "pbe_", "scan_"]
# shapes that could be mistaken for a per-atom array in a small cell
AMBIGUOUS = {"stress", "dipole", "magmom"}


def fix_prefix(prefix: str | None) -> str:
    """`mpa0` -> `mpa0_`; `_`/None -> `` (plain energy/forces)."""
    if not prefix or prefix in ("_", "none", "None", "-"):
        return ""
    return prefix if prefix.endswith("_") else prefix + "_"


def keys(prefix: str) -> tuple[str, str]:
    return f"{prefix}energy", f"{prefix}forces"


def has_labels(atoms: Atoms, prefix: str, forces: bool = True) -> bool:
    ekey, fkey = keys(prefix)
    return ekey in atoms.info and (not forces or fkey in atoms.arrays)


def read_dataset(infile: str, index: str = ":") -> list:
    frames = ase.io.read(infile, index=index)
    frames = [frames] if isinstance(frames, Atoms) else frames
    # ASE parks plain energy/forces/stress/... in a SinglePointCalculator instead of
    # info/arrays. Move everything across, or it is silently dropped on write.
    for atoms in frames:
        for key, value in (getattr(atoms.calc, "results", None) or {}).items():
            value = np.asarray(value)
            per_atom = value.ndim >= 1 and value.shape[0] == len(atoms) and key not in AMBIGUOUS
            if per_atom:
                atoms.arrays.setdefault(key, value)
            elif key not in atoms.info:
                atoms.info[key] = float(value) if value.ndim == 0 else value
        atoms.calc = None

    n_atoms = np.array([len(a) for a in frames])
    elements = sorted({s for a in frames for s in set(a.get_chemical_symbols())})
    n_pbc = sum(bool(np.any(a.pbc)) for a in frames)
    LOG(f"[key]{infile}[/key] · [hi]{len(frames)}[/hi] structures · {n_atoms.sum()} atoms "
        f"({n_atoms.min()}–{n_atoms.max()} per structure) · {' '.join(elements)} · {n_pbc} periodic")
    return frames


def find_ref(frames: list, exclude: str) -> str | None:
    """Guess the reference prefix (dft_energy / REF_energy / energy ...)."""
    for forces in (True, False):
        for cand in REF_CANDIDATES:
            if cand != exclude and has_labels(frames[0], cand, forces):
                return cand
    return None


def energies(frames: list, prefix: str) -> np.ndarray:
    return np.array([float(a.info[keys(prefix)[0]]) for a in frames])


def forces(frames: list, prefix: str) -> np.ndarray:
    return np.concatenate([np.asarray(a.arrays[keys(prefix)[1]]).ravel() for a in frames])

def composition(frames: list) -> tuple[np.ndarray, list]:
    """C[i, j] = how many atoms of element j structure i contains."""
    zs = sorted({int(z) for a in frames for z in a.numbers})
    return np.array([[int(np.count_nonzero(a.numbers == z)) for z in zs] for a in frames], float), zs

def group_of(frames: list, key: str | None) -> np.ndarray | None:
    """
    Per-structure group labels taken from an existing info key (config_type, step, ...)
    or from the virtual keys `formula` / `natoms`. A numeric key with more than 8 distinct
    values is cut into quartiles, so `-group mpa0_dE` or `-group temperature_K` also work.
    """
    if not key:
        return None
    if key == "formula":
        values = np.array([a.get_chemical_formula(mode="hill", empirical=True) for a in frames])
    elif key == "natoms":
        values = np.array([len(a) for a in frames])
    elif all(key in a.info for a in frames):
        values = np.array([a.info[key] for a in frames])
    else:
        missing = sum(key not in a.info for a in frames)
        LOG(f"[warn]-group {key}: missing from {missing}/{len(frames)} structures — not grouping[/warn]")
        return None

    if values.ndim > 1:
        LOG(f"[warn]-group {key}: values are arrays, not labels — not grouping[/warn]")
        return None
    if values.dtype.kind in "iuf" and len(np.unique(values)) > 8:
        edges = np.quantile(values.astype(float), [0.0, 0.25, 0.5, 0.75, 1.0])
        which = np.clip(np.searchsorted(edges[1:-1], values.astype(float), side="right"), 0, 3)
        values = np.array([f"Q{i + 1} [{edges[i]:.3g}, {edges[i + 1]:.3g}]" for i in which])
    return values.astype(str)

def free_prefix(frames: list, prefix: str, overwrite: bool = False) -> str:
    """Never silently overwrite an existing label set: mpa0_ -> mpa0_c1_ -> mpa0_c2_ ..."""
    if overwrite or not any(keys(prefix)[0] in a.info for a in frames):
        return prefix
    base = prefix.rstrip("_")
    for n in range(1, 100):
        candidate = f"{base}_c{n}_"
        if not any(keys(candidate)[0] in a.info for a in frames):
            LOG(f"[warn]{keys(prefix)[0]} is already in the file — writing "
                f"[key]{keys(candidate)[0]}[/key] instead (-overwrite to replace)[/warn]")
            return candidate
    return prefix
