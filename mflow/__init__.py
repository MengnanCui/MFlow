# -*- coding: utf-8 -*-
"""
MFlow — MACE toolkit: evaluation, relaxation, training and LAMMPS molecular dynamics.

    mflow calc  : evaluate an xyz dataset with MACE -> energies & forces
    mflow relax : optimise every structure with an ASE optimiser
    mflow plot  : re-analyse / re-plot an already evaluated file

Writes  <stem>_mace.xyz / <stem>_relax.xyz (labels <prefix>energy, <prefix>forces, ...),
        a summary png, a metrics json and py.log.
"""

__version__ = "0.7.0"

# the names tools/ and older scripts import from `mflow`
from .data import (REF_CANDIDATES, composition, energies, find_ref, fix_prefix, forces,  # noqa: E402,F401
                   free_prefix, group_of, has_labels, keys, read_dataset)
from .log import LOG, MONITOR, elapsed, num, table  # noqa: E402,F401
from .model import DEFAULT_MODEL, build_calculator, load_model, pick_device, resolve_model, versions  # noqa: E402,F401
from .style import COLORS, LINESTYLES, MARKERS, plt  # noqa: E402,F401

__all__ = ["COLORS", "DEFAULT_MODEL", "LINESTYLES", "LOG", "MARKERS", "MONITOR", "REF_CANDIDATES",
           "build_calculator", "composition", "elapsed", "energies", "find_ref", "fix_prefix", "forces",
           "free_prefix", "group_of", "has_labels", "keys", "load_model", "num", "pick_device", "plt",
           "read_dataset", "resolve_model", "table", "versions"]
