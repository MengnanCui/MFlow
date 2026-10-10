# -*- coding: utf-8 -*-
"""
MFlow — MACE toolkit: dataset checks, training, evaluation, LAMMPS molecular dynamics.

    mflow check / train / train-report          dataset health, mace_run_train + live monitoring, analysis
    mflow calc / relax / plot                   evaluation (incl. committees), ASE relaxation, re-plotting
    mflow deploy / lmp-check / md / md-report   ML-IAP export, LAMMPS probe, KOKKOS MD, quality gates

Every command writes a json report with status pass | warn | fail (see mflow.report).
"""

__version__ = "0.2.0a1"

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
