#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFlow — a single-file toolkit for machine-learning interatomic potentials.

Version 0.1.0 : static MACE evaluation (energy & forces) of an xyz dataset.

Everything lives in this one file on purpose, so it can be copied to a cluster
without installing anything.

--------------------------------------------------------------------------------
Quick start
--------------------------------------------------------------------------------

    # 1) evaluate a dataset with the default foundation model (mace-mp-0 medium)
    python mflow.py calc -i data.xyz

    # 2) pick a foundation model / batch size / output labels
    python mflow.py calc -i data.xyz -m medium-mpa-0 -b 32 -p mpa0_

    # 3) use your own fine-tuned model file
    python mflow.py calc -i data.xyz -m /path/to/MACE_model_swa.model

    # 4) compare against DFT labels already stored in the file (dft_energy/dft_forces)
    python mflow.py calc -i data.xyz --ref-prefix dft_

    # 5) only re-plot / re-analyse an already evaluated file
    python mflow.py plot -i data_mace.xyz --pred-prefix mpa0_ --ref-prefix dft_

Outputs
    <stem>_mace.xyz          structures + `<prefix>energy` / `<prefix>forces`
    py.log                   detailed log (timings, per-batch progress, RMSE ...)
    <stem>_metrics.json      machine readable summary
    *.png                    parity / error / distribution plots

--------------------------------------------------------------------------------
Author : Mengnan Cui
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import ase.io
from ase import Atoms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


__version__ = "0.1.0"


# ============================================================================= #
#
#                                   STYLE
#
# ============================================================================= #

COLORS = [
    "#2470a0",  # 1  muted blue
    "#ca3e47",  # 2  muted red
    "#f29c2b",  # 3  muted orange
    "#1f640a",  # 4  dark green
    "#2ca02c",  # 5  asparagus green
    "#9467bd",  # 6  muted purple
    "#8c564b",  # 7  chestnut brown
    "#e377c2",  # 8  raspberry pink
    "#7f7f7f",  # 9  middle gray
    "#bcbd22",  # 10 curry yellow-green
    "#17becf",  # 11 blue-teal
    "#005555",  # 12 FHI green
]

MARKERS = ["o", "v", "s", "*", "p", "P", ","]

LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
    (0, (5, 5)),  # long dashed
]

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.linewidth": 2,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.minor.width": 1.5,
        "ytick.minor.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "lines.linewidth": 2,
        "lines.markersize": 13,
        "lines.markeredgewidth": 2,
        "legend.fontsize": 10,
        "legend.markerscale": 1.5,
        "legend.framealpha": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ============================================================================= #
#
#                                  LOGGING
#
# ============================================================================= #

LOGGER_NAME = "mflow"
log = logging.getLogger(LOGGER_NAME)

# points above which a parity plot switches from scatter to hexbin
HEXBIN_THRESHOLD = 20_000
# points above which a parity plot is subsampled before drawing
PLOT_MAX_POINTS = 500_000


def setup_logger(log_file: str = "py.log", level: str = "INFO", append: bool = False):
    """
    Two channels:
      * console -> clean, only the things you want to see while it runs
      * file    -> everything, with timestamps, function names and DEBUG detail
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a" if append else "w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(funcName)-22s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    # matplotlib/PIL are very chatty at DEBUG level
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def banner(title: str, char: str = "=", width: int = 78):
    log.info("")
    log.info(char * width)
    log.info(f"  {title}")
    log.info(char * width)


def log_params(params: dict, title: str = "Input parameters"):
    """Pretty print the key parameters (console + log file)."""
    banner(title)
    width = max(len(k) for k in params) if params else 0
    for key, value in params.items():
        log.info(f"  {key:<{width}} : {value}")
    log.info("")


def fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{int(seconds // 60)} min {seconds % 60:.0f} s"
    return f"{int(seconds // 3600)} h {int((seconds % 3600) // 60)} min"


# ============================================================================= #
#
#                                 DATA HELPERS
#
# ============================================================================= #

# candidate prefixes probed when --ref-prefix is not given
REF_PREFIX_CANDIDATES = ["", "REF_", "ref_", "dft_", "DFT_", "pbe_", "scan_"]

# convenient short names for MACE foundation models
MODEL_ALIASES = {
    "mp": "medium",
    "mp0": "medium",
    "mp-0": "medium",
    "mpa": "medium-mpa-0",
    "mpa0": "medium-mpa-0",
    "mpa-0": "medium-mpa-0",
    "omat": "medium-omat-0",
    "omat0": "medium-omat-0",
    "omat-0": "medium-omat-0",
}


def normalise_prefix(prefix: str) -> str:
    """`mpa0` -> `mpa0_`, `_` / `None` -> `''` (i.e. plain `energy`/`forces`)."""
    if prefix is None:
        return ""
    if prefix in ("_", "none", "None", "-"):
        return ""
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    return prefix


def energy_key(prefix: str) -> str:
    return f"{prefix}energy"


def forces_key(prefix: str) -> str:
    return f"{prefix}forces"


def has_labels(atoms: Atoms, prefix: str, need_forces: bool = True) -> bool:
    ok = energy_key(prefix) in atoms.info
    if need_forces:
        ok = ok and (forces_key(prefix) in atoms.arrays)
    return ok


def detect_ref_prefix(atoms_list: list, exclude: str) -> str | None:
    """
    Guess which prefix holds the reference (DFT) labels, e.g. `dft_energy` +
    `dft_forces`. Returns None when nothing usable is found.
    """
    probe = atoms_list[0]
    for cand in REF_PREFIX_CANDIDATES:
        if cand == exclude:
            continue
        if has_labels(probe, cand, need_forces=True):
            return cand
    # energy-only reference is still useful
    for cand in REF_PREFIX_CANDIDATES:
        if cand == exclude:
            continue
        if has_labels(probe, cand, need_forces=False):
            return cand
    return None


def promote_calculator_results(atoms_list: list):
    """
    extxyz files that store the reference as plain `energy`/`forces` end up in
    a SinglePointCalculator instead of info/arrays. Move them back so that the
    unprefixed labels can be used as a reference like any other prefix.
    """
    moved = 0
    for atoms in atoms_list:
        results = getattr(atoms.calc, "results", None) or {}
        if "energy" in results and "energy" not in atoms.info:
            atoms.info["energy"] = float(results["energy"])
            moved += 1
        if "forces" in results and "forces" not in atoms.arrays:
            atoms.arrays["forces"] = np.asarray(results["forces"])
        atoms.calc = None
    if moved:
        log.info(f"Promoted the attached calculator results of {moved} structures to 'energy'/'forces'")


def read_dataset(infile: str, index: str = ":") -> list:
    t0 = time.perf_counter()
    log.debug(f"Reading {infile} with index='{index}'")
    atoms_list = ase.io.read(infile, index=index)
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]
    promote_calculator_results(atoms_list)
    dt = time.perf_counter() - t0
    n_at = sum(len(a) for a in atoms_list)
    log.info(f"Read {len(atoms_list)} structures ({n_at} atoms) from {infile} in {fmt_time(dt)}")
    return atoms_list


def dataset_summary(atoms_list: list) -> dict:
    n_atoms = np.array([len(a) for a in atoms_list])
    symbols = sorted({s for a in atoms_list for s in set(a.get_chemical_symbols())})
    pbc = sum(1 for a in atoms_list if bool(np.any(a.pbc)))
    return {
        "n_structures": int(len(atoms_list)),
        "n_atoms_total": int(n_atoms.sum()),
        "n_atoms_min": int(n_atoms.min()),
        "n_atoms_max": int(n_atoms.max()),
        "n_atoms_mean": float(n_atoms.mean()),
        "elements": symbols,
        "n_periodic": int(pbc),
    }


def gather_forces(atoms_list: list, prefix: str) -> np.ndarray:
    """Flatten all force components of the dataset into a 1D array."""
    key = forces_key(prefix)
    return np.concatenate([np.asarray(a.arrays[key]).reshape(-1) for a in atoms_list])


def gather_energies(atoms_list: list, prefix: str) -> np.ndarray:
    key = energy_key(prefix)
    return np.array([float(np.asarray(a.info[key]).reshape(-1)[0]) for a in atoms_list])


# ============================================================================= #
#
#                              MACE MODEL LOADING
#
# ============================================================================= #


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing
        pass
    return "cpu"


def resolve_model_spec(spec: str) -> tuple[str, str]:
    """
    Returns (kind, value) with kind in {'file', 'foundation'}.
    A local file always wins over an alias.
    """
    if spec and os.path.isfile(os.path.expanduser(spec)):
        return "file", os.path.expanduser(spec)
    return "foundation", MODEL_ALIASES.get(str(spec).lower(), spec)


def torch_load(path: str, device):
    """torch.load that works before and after the torch>=2.6 weights_only switch."""
    import torch

    try:
        return torch.load(f=path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(f=path, map_location=device)


def load_raw_model(model_spec: str, device: str, default_dtype: str):
    """
    Load a MACE model as a plain torch module (needed for batched evaluation).
    Works both for a local `.model` file and for a foundation model name.
    """
    import torch
    from mace.tools import torch_tools

    torch_tools.set_default_dtype(default_dtype)
    dev = torch_tools.init_device(device)

    kind, value = resolve_model_spec(model_spec)
    t0 = time.perf_counter()
    if kind == "file":
        log.info(f"Loading local MACE model : {value}")
        model = torch_load(value, dev)
    else:
        log.info(f"Loading MACE foundation model : {value} (downloaded/cached by mace)")
        try:
            from mace.calculators import mace_mp
        except ImportError:  # pragma: no cover
            from mace.calculators.foundations_models import mace_mp
        model = mace_mp(
            model=value,
            device=str(dev),
            default_dtype=default_dtype,
            return_raw_model=True,
        )

    model = model.to(dev)
    model = model.double() if default_dtype == "float64" else model.float()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    log.info(f"Model ready on '{dev}' in {fmt_time(time.perf_counter() - t0)}")
    log.debug(f"r_max = {float(model.r_max)}, atomic numbers = {model.atomic_numbers.tolist()}")
    n_par = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters : {n_par:,}  |  r_max = {float(model.r_max):.2f} A")
    return model, dev


def build_ase_calculator(model_spec: str, device: str, default_dtype: str, head: str | None):
    """Fallback engine: the plain ASE calculator (one structure at a time)."""
    kind, value = resolve_model_spec(model_spec)
    kwargs = {"device": device, "default_dtype": default_dtype}
    if head:
        kwargs["head"] = head

    if kind == "file":
        from mace.calculators import MACECalculator

        log.info(f"Loading local MACE model (ASE engine) : {value}")
        try:
            return MACECalculator(model_paths=value, **kwargs)
        except TypeError:
            return MACECalculator(models=value, **kwargs)

    try:
        from mace.calculators import mace_mp
    except ImportError:  # pragma: no cover
        from mace.calculators.foundations_models import mace_mp

    log.info(f"Loading MACE foundation model (ASE engine) : {value}")
    return mace_mp(model=value, **kwargs)


# ============================================================================= #
#
#                               MACE EVALUATION
#
# ============================================================================= #


def _clean_copy(atoms: Atoms, head: str | None = None) -> Atoms:
    """
    Geometry-only copy. Keeping the original info/arrays out of the way avoids
    clashes between the reference labels and MACE's own key parsing.
    """
    new = Atoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )
    if head is not None:
        new.info["head"] = head
    return new


def _config_from_atoms(data_mod, atoms: Atoms):
    """`config_from_atoms` changed signature across mace versions."""
    try:
        return data_mod.config_from_atoms(atoms)
    except TypeError:
        from mace.data.utils import KeySpecification  # type: ignore

        return data_mod.config_from_atoms(atoms, key_specification=KeySpecification())


def _atomic_data_from_config(data_mod, config, z_table, cutoff, heads):
    """`AtomicData.from_config` gained a `heads` argument in newer mace versions."""
    try:
        return data_mod.AtomicData.from_config(
            config, z_table=z_table, cutoff=cutoff, heads=heads
        )
    except TypeError:
        return data_mod.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)


def mace_eval_batch(
    atoms_list: list,
    model_spec: str,
    batch_size: int = 32,
    device: str = "auto",
    default_dtype: str = "float64",
    prefix: str = "mpa0_",
    head: str | None = None,
) -> list:
    """
    Batched static evaluation: energies (eV) and forces (eV/A) are written back
    into `atoms.info[prefix + 'energy']` and `atoms.arrays[prefix + 'forces']`.
    """
    import torch
    import mace.data as mace_data
    from mace.tools import torch_geometric, torch_tools, utils

    model, dev = load_raw_model(model_spec, device, default_dtype)

    t0 = time.perf_counter()
    configs = [_config_from_atoms(mace_data, _clean_copy(a, head)) for a in atoms_list]
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", None)
    if head is not None and heads is not None and head not in heads:
        log.warning(f"Head '{head}' not in model heads {heads}; falling back to the default head")

    dataset = [
        _atomic_data_from_config(
            mace_data, config, z_table=z_table, cutoff=float(model.r_max), heads=heads
        )
        for config in configs
    ]
    log.info(f"Built neighbour lists for {len(dataset)} structures in {fmt_time(time.perf_counter() - t0)}")

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    n_batches = len(data_loader)
    log_every = max(1, n_batches // 20)
    log.info(f"Evaluating {len(atoms_list)} structures in {n_batches} batches (batch_size = {batch_size})")

    energies_list, forces_collection = [], []
    t_start = time.perf_counter()
    for i, batch in enumerate(data_loader, start=1):
        batch = batch.to(dev)
        out = model(batch.to_dict(), compute_stress=False)
        energies_list.append(torch_tools.to_numpy(out["energy"]))

        forces = np.split(
            torch_tools.to_numpy(out["forces"]),
            indices_or_sections=batch.ptr[1:].cpu().numpy(),
            axis=0,
        )
        forces_collection.append(forces[:-1])  # last split is empty

        if i % log_every == 0 or i == n_batches:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (n_batches - i)
            log.info(
                f"  batch {i:>5}/{n_batches}  ({100.0 * i / n_batches:5.1f} %)  "
                f"elapsed {fmt_time(elapsed)}  eta {fmt_time(eta)}"
            )

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [f for chunk in forces_collection for f in chunk]
    assert len(atoms_list) == len(energies) == len(forces_list), (
        f"size mismatch: {len(atoms_list)} structures, {len(energies)} energies, "
        f"{len(forces_list)} force sets"
    )

    for atoms, energy, forces in zip(atoms_list, energies, forces_list):
        atoms.calc = None  # crucial, otherwise ase re-writes its own keys
        atoms.info[energy_key(prefix)] = float(energy)
        atoms.arrays[forces_key(prefix)] = np.asarray(forces)

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    wall = time.perf_counter() - t_start
    n_at = sum(len(a) for a in atoms_list)
    log.info(
        f"Batched evaluation finished in {fmt_time(wall)}  "
        f"({len(atoms_list) / max(wall, 1e-9):.1f} struct/s, {n_at / max(wall, 1e-9):.0f} atom/s)"
    )
    return atoms_list


def mace_eval_ase(
    atoms_list: list,
    model_spec: str,
    device: str = "auto",
    default_dtype: str = "float64",
    prefix: str = "mpa0_",
    head: str | None = None,
) -> list:
    """Structure-by-structure evaluation through the ASE calculator interface."""
    calc = build_ase_calculator(model_spec, device, default_dtype, head)

    n = len(atoms_list)
    log_every = max(1, n // 20)
    log.info(f"Evaluating {n} structures one by one (ASE engine)")

    t_start = time.perf_counter()
    for i, atoms in enumerate(atoms_list, start=1):
        work = _clean_copy(atoms, head)
        work.calc = calc
        energy = float(work.get_potential_energy())
        forces = np.asarray(work.get_forces())

        atoms.calc = None
        atoms.info[energy_key(prefix)] = energy
        atoms.arrays[forces_key(prefix)] = forces

        if i % log_every == 0 or i == n:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (n - i)
            log.info(
                f"  structure {i:>6}/{n}  ({100.0 * i / n:5.1f} %)  "
                f"elapsed {fmt_time(elapsed)}  eta {fmt_time(eta)}"
            )

    wall = time.perf_counter() - t_start
    log.info(f"ASE evaluation finished in {fmt_time(wall)} ({n / max(wall, 1e-9):.1f} struct/s)")
    return atoms_list


def run_mace(
    atoms_list: list,
    model_spec: str,
    engine: str = "auto",
    batch_size: int = 32,
    device: str = "auto",
    default_dtype: str = "float64",
    prefix: str = "mpa0_",
    head: str | None = None,
) -> list:
    """Dispatch to the batched engine, falling back to ASE when it breaks."""
    if engine in ("auto", "batch"):
        try:
            return mace_eval_batch(
                atoms_list,
                model_spec,
                batch_size=batch_size,
                device=device,
                default_dtype=default_dtype,
                prefix=prefix,
                head=head,
            )
        except Exception as exc:
            if engine == "batch":
                raise
            log.warning(f"Batched engine failed ({type(exc).__name__}: {exc})")
            log.warning("Falling back to the ASE engine — see the log file for the traceback")
            log.debug(traceback.format_exc())

    return mace_eval_ase(
        atoms_list,
        model_spec,
        device=device,
        default_dtype=default_dtype,
        prefix=prefix,
        head=head,
    )


# ============================================================================= #
#
#                                   METRICS
#
# ============================================================================= #


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def prediction_summary(atoms_list: list, prefix: str) -> dict:
    """Basic statistics of the prediction itself (no reference needed)."""
    e = gather_energies(atoms_list, prefix)
    n_atoms = np.array([len(a) for a in atoms_list])
    e_pa = e / n_atoms

    key = forces_key(prefix)
    fmax = np.array([np.linalg.norm(a.arrays[key], axis=1).max() for a in atoms_list])
    f_all = gather_forces(atoms_list, prefix)

    return {
        "energy_total_eV": {
            "min": float(e.min()),
            "max": float(e.max()),
            "mean": float(e.mean()),
        },
        "energy_per_atom_eV": {
            "min": float(e_pa.min()),
            "max": float(e_pa.max()),
            "mean": float(e_pa.mean()),
            "std": float(e_pa.std()),
        },
        "forces_eV_per_A": {
            "component_min": float(f_all.min()),
            "component_max": float(f_all.max()),
            "component_rms": float(np.sqrt(np.mean(f_all**2))),
            "fmax_mean": float(fmax.mean()),
            "fmax_max": float(fmax.max()),
        },
    }


def compute_errors(atoms_list: list, pred_prefix: str, ref_prefix: str) -> dict:
    """
    RMSE / MAE of the prediction against the reference labels.

    Energies are compared per atom. Because different codes use different
    atomic references, the mean-shifted error is reported as well — that is the
    number to look at when the two methods differ by a constant offset.
    """
    n_atoms = np.array([len(a) for a in atoms_list], dtype=float)
    e_pred = gather_energies(atoms_list, pred_prefix)
    e_ref = gather_energies(atoms_list, ref_prefix)

    de = e_pred - e_ref
    de_pa = de / n_atoms
    de_pa_shift = de_pa - de_pa.mean()

    errors = {
        "n_structures": int(len(atoms_list)),
        "energy": {
            "MAE_meV_per_atom": float(np.mean(np.abs(de_pa)) * 1000),
            "RMSE_meV_per_atom": float(np.sqrt(np.mean(de_pa**2)) * 1000),
            "MAE_shifted_meV_per_atom": float(np.mean(np.abs(de_pa_shift)) * 1000),
            "RMSE_shifted_meV_per_atom": float(np.sqrt(np.mean(de_pa_shift**2)) * 1000),
            "mean_offset_meV_per_atom": float(de_pa.mean() * 1000),
            "MAE_eV": float(np.mean(np.abs(de))),
            "RMSE_eV": float(np.sqrt(np.mean(de**2))),
            "R2": r_squared(e_ref / n_atoms, e_pred / n_atoms),
        },
    }

    if forces_key(ref_prefix) in atoms_list[0].arrays:
        f_pred = gather_forces(atoms_list, pred_prefix)
        f_ref = gather_forces(atoms_list, ref_prefix)
        df = f_pred - f_ref
        errors["forces"] = {
            "n_components": int(df.size),
            "MAE_meV_per_A": float(np.mean(np.abs(df)) * 1000),
            "RMSE_meV_per_A": float(np.sqrt(np.mean(df**2)) * 1000),
            "max_abs_error_meV_per_A": float(np.max(np.abs(df)) * 1000),
            "R2": r_squared(f_ref, f_pred),
        }
    else:
        log.warning(f"No '{forces_key(ref_prefix)}' in the dataset — skipping force errors")

    return errors


def log_errors(errors: dict, pred_prefix: str, ref_prefix: str):
    banner(f"Error : {pred_prefix or 'pred'} vs {ref_prefix or 'ref'}  (n = {errors['n_structures']})")
    e = errors["energy"]
    log.info("  Energy")
    log.info(f"    MAE           : {e['MAE_meV_per_atom']:10.3f} meV/atom")
    log.info(f"    RMSE          : {e['RMSE_meV_per_atom']:10.3f} meV/atom")
    log.info(f"    MAE  (shifted): {e['MAE_shifted_meV_per_atom']:10.3f} meV/atom")
    log.info(f"    RMSE (shifted): {e['RMSE_shifted_meV_per_atom']:10.3f} meV/atom")
    log.info(f"    mean offset   : {e['mean_offset_meV_per_atom']:10.3f} meV/atom")
    log.info(f"    R^2           : {e['R2']:10.4f}")
    if "forces" in errors:
        f = errors["forces"]
        log.info("  Forces")
        log.info(f"    MAE           : {f['MAE_meV_per_A']:10.3f} meV/A")
        log.info(f"    RMSE          : {f['RMSE_meV_per_A']:10.3f} meV/A")
        log.info(f"    max |error|   : {f['max_abs_error_meV_per_A']:10.3f} meV/A")
        log.info(f"    R^2           : {f['R2']:10.4f}")
    log.info("")


def write_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    log.info(f"Metrics written to {path}")


# ============================================================================= #
#
#                                 VISUALISATION
#
# ============================================================================= #


def _subsample(x: np.ndarray, y: np.ndarray, max_points: int = PLOT_MAX_POINTS):
    if x.size <= max_points:
        return x, y, False
    rng = np.random.default_rng(seed=8465)
    idx = rng.choice(x.size, size=max_points, replace=False)
    log.info(f"  plotting a random subsample of {max_points:,} / {x.size:,} points")
    return x[idx], y[idx], True


def plot_parity(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    outfile: str,
    annotation: str = "",
):
    """Reference (x) vs prediction (y) with the y = x guide line."""
    x, y, _ = _subsample(np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    margin = (hi - lo) * 0.05 if hi > lo else 1.0
    lo, hi = lo - margin, hi + margin

    fig, ax = plt.subplots(figsize=(5, 4))

    if x.size > HEXBIN_THRESHOLD:
        from matplotlib.colors import LogNorm

        hb = ax.hexbin(x, y, gridsize=(60, 60), cmap="viridis", mincnt=1, norm=LogNorm())
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("counts", fontsize=13)
    else:
        # keep the markers readable when thousands of points overlap
        size = 80 if x.size <= 500 else (40 if x.size <= 3000 else 12)
        ax.scatter(
            x,
            y,
            color=COLORS[0],
            marker=MARKERS[0],
            alpha=0.7,
            s=size,
            edgecolors=COLORS[0],
            linewidths=1.5 if size >= 40 else 0.5,
            zorder=3,
        )

    ax.plot([lo, hi], [lo, hi], color="black", linestyle=LINESTYLES[1], linewidth=2, zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotation:
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6, edgecolor="none"),
        )
    ax.tick_params(axis="both", which="major", length=6, width=2, direction="in", labelsize=13)
    ax.tick_params(axis="both", which="minor", length=3, width=1.5, direction="in")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    log.info(f"  figure -> {outfile}")


def plot_error_hist(
    energy_err: np.ndarray,
    force_err: np.ndarray | None,
    outfile: str,
):
    """Distribution of the signed errors (energy per atom / force components)."""
    n_panels = 1 if force_err is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    axes = np.atleast_1d(axes)

    axes[0].hist(
        energy_err,
        bins=50,
        color=COLORS[0],
        alpha=0.75,
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )
    axes[0].set_xlabel("Energy error (meV/atom)")
    axes[0].set_ylabel("Counts")
    axes[0].set_title("Energy")

    if force_err is not None:
        fe, _, _ = _subsample(force_err, force_err)
        axes[1].hist(
            fe,
            bins=50,
            color=COLORS[1],
            alpha=0.75,
            edgecolor="black",
            linewidth=1.0,
            zorder=2,
        )
        axes[1].set_xlabel(r"Force error (meV/$\mathrm{\AA}$)")
        axes[1].set_ylabel("Counts")
        axes[1].set_title("Forces")

    for ax in axes:
        ax.axvline(0.0, color="black", linestyle=LINESTYLES[1], linewidth=2, zorder=3)
        ax.tick_params(axis="both", which="major", length=6, width=2, direction="in", labelsize=13)
        ax.tick_params(axis="both", which="minor", length=3, width=1.5, direction="in")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    log.info(f"  figure -> {outfile}")


def plot_distribution(atoms_list: list, prefix: str, outfile: str):
    """Predicted energy-per-atom and |F| distributions (used when no reference)."""
    e = gather_energies(atoms_list, prefix)
    n_atoms = np.array([len(a) for a in atoms_list], dtype=float)
    e_pa = e / n_atoms
    fmax = np.array(
        [np.linalg.norm(a.arrays[forces_key(prefix)], axis=1).max() for a in atoms_list]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(e_pa, bins=50, color=COLORS[0], alpha=0.75, edgecolor="black", linewidth=1.0, zorder=2)
    axes[0].set_xlabel("Energy (eV/atom)")
    axes[0].set_ylabel("Counts")
    axes[0].set_title(f"{prefix or 'pred'}energy")

    axes[1].hist(fmax, bins=50, color=COLORS[1], alpha=0.75, edgecolor="black", linewidth=1.0, zorder=2)
    axes[1].set_xlabel(r"max |F| per structure (eV/$\mathrm{\AA}$)")
    axes[1].set_ylabel("Counts")
    axes[1].set_title(f"{prefix or 'pred'}forces")

    for ax in axes:
        ax.tick_params(axis="both", which="major", length=6, width=2, direction="in", labelsize=13)
        ax.tick_params(axis="both", which="minor", length=3, width=1.5, direction="in")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    log.info(f"  figure -> {outfile}")


def make_plots(
    atoms_list: list,
    pred_prefix: str,
    ref_prefix: str | None,
    errors: dict | None,
    outdir: str,
    stem: str,
):
    banner("Visualisation")
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    pred_tag = (pred_prefix or "pred").rstrip("_").upper()

    if ref_prefix is None or errors is None:
        plot_distribution(atoms_list, pred_prefix, str(outdir_p / f"{stem}_distribution.png"))
        log.info("No reference labels — only the prediction distributions were plotted")
        return

    ref_tag = (ref_prefix or "ref").rstrip("_").upper() or "REF"

    # ---- energy parity (per atom, both series shifted by their own mean) ----
    n_atoms = np.array([len(a) for a in atoms_list], dtype=float)
    e_pred = gather_energies(atoms_list, pred_prefix) / n_atoms
    e_ref = gather_energies(atoms_list, ref_prefix) / n_atoms
    e_pred_s = e_pred - e_pred.mean()
    e_ref_s = e_ref - e_ref.mean()

    ee = errors["energy"]
    plot_parity(
        e_ref_s,
        e_pred_s,
        xlabel=f"{ref_tag} rel. energy (eV/atom)",
        ylabel=f"{pred_tag} rel. energy (eV/atom)",
        title="Energy",
        outfile=str(outdir_p / f"{stem}_energy_parity.png"),
        annotation=(
            f"RMSE = {ee['RMSE_shifted_meV_per_atom']:.2f} meV/atom\n"
            f"R$^2$ = {ee['R2']:.4f}\nN = {len(e_ref)}"
        ),
    )

    # ---- forces parity (all components) ----
    force_err = None
    if "forces" in errors:
        f_pred = gather_forces(atoms_list, pred_prefix)
        f_ref = gather_forces(atoms_list, ref_prefix)
        force_err = (f_pred - f_ref) * 1000
        fe = errors["forces"]
        plot_parity(
            f_ref,
            f_pred,
            xlabel=rf"{ref_tag} force comp. (eV/$\mathrm{{\AA}}$)",
            ylabel=rf"{pred_tag} force comp. (eV/$\mathrm{{\AA}}$)",
            title="Forces",
            outfile=str(outdir_p / f"{stem}_forces_parity.png"),
            annotation=(
                f"RMSE = {fe['RMSE_meV_per_A']:.2f} meV/$\\mathrm{{\\AA}}$\n"
                f"R$^2$ = {fe['R2']:.4f}\nN = {fe['n_components']}"
            ),
        )

    # ---- error histograms ----
    energy_err = (e_pred - e_ref) * 1000
    energy_err = energy_err - energy_err.mean()
    plot_error_hist(energy_err, force_err, str(outdir_p / f"{stem}_error_hist.png"))


# ============================================================================= #
#
#                                   COMMANDS
#
# ============================================================================= #


def environment_info() -> dict:
    info = {
        "mflow_version": __version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "cwd": os.getcwd(),
        "numpy": np.__version__,
    }
    try:
        import ase

        info["ase"] = ase.__version__
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "not installed"
    try:
        import mace

        info["mace"] = getattr(mace, "__version__", "unknown")
    except Exception:
        info["mace"] = "not installed"
    return info


def cmd_calc(args):
    t_total = time.perf_counter()

    prefix = normalise_prefix(args.prefix)
    infile = args.input
    stem = Path(infile).stem
    outfile = args.output or str(Path(args.outdir) / f"{stem}_mace.xyz")
    device = resolve_device(args.device)
    kind, model_value = resolve_model_spec(args.model)

    banner(f"MFlow v{__version__}  |  static MACE evaluation of energies and forces")
    log.info(f"  started at {datetime.now():%Y-%m-%d %H:%M:%S}")

    log_params(
        {
            "input": infile,
            "index": args.index,
            "model": f"{model_value}  ({'local file' if kind == 'file' else 'foundation model'})",
            "batch_size": args.batch_size,
            "engine": args.engine,
            "device": f"{device} (requested: {args.device})",
            "default_dtype": args.dtype,
            "head": args.head or "-",
            "output prefix": f"{energy_key(prefix)} / {forces_key(prefix)}",
            "output xyz": outfile,
            "reference prefix": args.ref_prefix if args.ref_prefix is not None else "auto-detect",
            "outdir": args.outdir,
            "log file": args.log,
            "plots": "off" if args.no_plot else "on",
        }
    )
    log.info("Environment:")
    for key, value in environment_info().items():
        log.info(f"  {key:<14} : {value}")
    log.info("")

    # ------------------------------------------------------------------ data
    banner("Dataset")
    atoms_list = read_dataset(infile, args.index)
    summary = dataset_summary(atoms_list)
    for key, value in summary.items():
        log.info(f"  {key:<14} : {value}")
    log.info("")

    # ------------------------------------------------------------- reference
    if args.ref_prefix is None:
        ref_prefix = detect_ref_prefix(atoms_list, exclude=prefix)
        if ref_prefix is None:
            log.info("No reference labels found in the input — running prediction only")
        else:
            log.info(f"Auto-detected reference labels: '{energy_key(ref_prefix)}'")
    else:
        ref_prefix = normalise_prefix(args.ref_prefix)
        if not has_labels(atoms_list[0], ref_prefix, need_forces=False):
            log.warning(f"'{energy_key(ref_prefix)}' not found in the input — no error analysis")
            ref_prefix = None

    # ------------------------------------------------------------ evaluation
    banner("MACE evaluation")
    atoms_list = run_mace(
        atoms_list,
        args.model,
        engine=args.engine,
        batch_size=args.batch_size,
        device=device,
        default_dtype=args.dtype,
        prefix=prefix,
        head=args.head,
    )

    # ---------------------------------------------------------------- output
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ase.io.write(outfile, atoms_list, format="extxyz")
    log.info(f"Wrote {len(atoms_list)} structures to {outfile} in {fmt_time(time.perf_counter() - t0)}")

    # -------------------------------------------------------------- analysis
    banner("Prediction summary")
    pred_stats = prediction_summary(atoms_list, prefix)
    log.info(f"  energy/atom : min {pred_stats['energy_per_atom_eV']['min']:.4f}  "
             f"max {pred_stats['energy_per_atom_eV']['max']:.4f}  "
             f"mean {pred_stats['energy_per_atom_eV']['mean']:.4f} eV/atom")
    log.info(f"  |F| max     : mean {pred_stats['forces_eV_per_A']['fmax_mean']:.4f}  "
             f"max {pred_stats['forces_eV_per_A']['fmax_max']:.4f} eV/A")
    log.info("")

    errors = None
    if ref_prefix is not None:
        errors = compute_errors(atoms_list, prefix, ref_prefix)
        log_errors(errors, prefix, ref_prefix)

    metrics = {
        "mflow_version": __version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": os.path.abspath(infile),
        "output": os.path.abspath(outfile),
        "model": model_value,
        "model_kind": kind,
        "batch_size": args.batch_size,
        "device": device,
        "default_dtype": args.dtype,
        "prediction_prefix": prefix,
        "reference_prefix": ref_prefix,
        "dataset": summary,
        "prediction": pred_stats,
        "errors": errors,
        "environment": environment_info(),
    }
    write_json(str(Path(args.outdir) / f"{stem}_metrics.json"), metrics)

    # ----------------------------------------------------------------- plots
    if not args.no_plot:
        make_plots(atoms_list, prefix, ref_prefix, errors, args.outdir, stem)

    banner("Done")
    log.info(f"  total wall time : {fmt_time(time.perf_counter() - t_total)}")
    log.info(f"  output xyz      : {outfile}")
    log.info(f"  log file        : {args.log}")
    log.info("")


def cmd_plot(args):
    banner(f"MFlow v{__version__}  |  analysis and plots")

    pred_prefix = normalise_prefix(args.pred_prefix)
    stem = Path(args.input).stem

    log_params(
        {
            "input": args.input,
            "index": args.index,
            "prediction prefix": energy_key(pred_prefix),
            "reference prefix": args.ref_prefix if args.ref_prefix is not None else "auto-detect",
            "outdir": args.outdir,
            "log file": args.log,
        }
    )

    atoms_list = read_dataset(args.input, args.index)
    if not has_labels(atoms_list[0], pred_prefix, need_forces=False):
        log.error(f"'{energy_key(pred_prefix)}' not found in {args.input} — nothing to plot")
        return 1

    if args.ref_prefix is None:
        ref_prefix = detect_ref_prefix(atoms_list, exclude=pred_prefix)
    else:
        ref_prefix = normalise_prefix(args.ref_prefix)
        if not has_labels(atoms_list[0], ref_prefix, need_forces=False):
            log.warning(f"'{energy_key(ref_prefix)}' not found — plotting distributions only")
            ref_prefix = None

    errors = None
    if ref_prefix is not None:
        log.info(f"Reference labels : '{energy_key(ref_prefix)}'")
        errors = compute_errors(atoms_list, pred_prefix, ref_prefix)
        log_errors(errors, pred_prefix, ref_prefix)
        write_json(str(Path(args.outdir) / f"{stem}_metrics.json"), errors)

    make_plots(atoms_list, pred_prefix, ref_prefix, errors, args.outdir, stem)
    banner("Done")
    return 0


# ============================================================================= #
#
#                                     CLI
#
# ============================================================================= #

EPILOG = """
examples
--------
  # default: mace-mp-0 medium, batch size 32, labels mpa0_energy / mpa0_forces
  python mflow.py calc -i data.xyz

  # a different foundation model and a bigger batch
  python mflow.py calc -i data.xyz -m medium-mpa-0 -b 64

  # your own model file, on cpu, single precision
  python mflow.py calc -i data.xyz -m ./MACE_model_swa.model --device cpu --dtype float32

  # compare with the DFT labels stored in the same file
  python mflow.py calc -i data.xyz --ref-prefix dft_

  # re-analyse an evaluated file without recomputing
  python mflow.py plot -i data_mace.xyz --pred-prefix mpa0_ --ref-prefix dft_

foundation model names
----------------------
  small | medium | large | medium-mpa-0 | medium-omat-0 | ...
  aliases: mpa -> medium-mpa-0, omat -> medium-omat-0, mp -> medium
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mflow.py",
        description=f"MFlow v{__version__} — MACE energies and forces for an xyz dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("-v", "--version", action="version", version=f"MFlow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ calc
    calc = sub.add_parser(
        "calc",
        help="run a static MACE calculation on an xyz dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    g_in = calc.add_argument_group("input")
    g_in.add_argument("-i", "--input", required=True, help="input xyz/extxyz file")
    g_in.add_argument("--index", default=":", help="ase slice of the input, e.g. ':100' (default: ':')")

    g_model = calc.add_argument_group("model")
    g_model.add_argument(
        "-m",
        "--model",
        default="medium",
        help="foundation model name (small/medium/large/medium-mpa-0/...) "
        "or a path to your own .model file (default: medium)",
    )
    g_model.add_argument("--model-path", dest="model", help="alias of --model")
    g_model.add_argument("--head", default=None, help="head name for a multi-head model")

    g_run = calc.add_argument_group("run")
    g_run.add_argument("-b", "--batch-size", type=int, default=32, help="batch size (default: 32)")
    g_run.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "batch", "ase"],
        help="'batch' = fast batched torch loop, 'ase' = one structure at a time, "
        "'auto' = batch with an ase fallback (default: auto)",
    )
    g_run.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="default: auto"
    )
    g_run.add_argument(
        "--dtype", default="float64", choices=["float64", "float32"], help="default: float64"
    )

    g_out = calc.add_argument_group("output")
    g_out.add_argument(
        "-p",
        "--prefix",
        default="mpa0_",
        help="prefix of the written labels, i.e. <prefix>energy and <prefix>forces "
        "(default: mpa0_)",
    )
    g_out.add_argument("-o", "--output", default=None, help="output xyz (default: <stem>_mace.xyz)")
    g_out.add_argument("--outdir", default=".", help="directory for xyz/json/png (default: .)")
    g_out.add_argument(
        "--ref-prefix",
        default=None,
        help="prefix of the reference labels used for RMSE/plots, e.g. 'dft_'. "
        "Auto-detected when omitted",
    )
    g_out.add_argument("--no-plot", action="store_true", help="skip the figures")
    g_out.add_argument("--log", default="py.log", help="log file (default: py.log)")
    g_out.add_argument("--log-append", action="store_true", help="append to the log instead of overwriting")
    g_out.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"], help="console level"
    )
    calc.set_defaults(func=cmd_calc)

    # ------------------------------------------------------------------ plot
    plot = sub.add_parser("plot", help="analyse/plot an already evaluated xyz file")
    plot.add_argument("-i", "--input", required=True, help="xyz file containing both label sets")
    plot.add_argument("--index", default=":", help="ase slice of the input (default: ':')")
    plot.add_argument("--pred-prefix", default="mpa0_", help="prediction prefix (default: mpa0_)")
    plot.add_argument("--ref-prefix", default=None, help="reference prefix, auto-detected when omitted")
    plot.add_argument("--outdir", default=".", help="directory for the figures (default: .)")
    plot.add_argument("--log", default="py.log", help="log file (default: py.log)")
    plot.add_argument("--log-append", action="store_true", help="append to the log")
    plot.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"], help="console level"
    )
    plot.set_defaults(func=cmd_plot)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logger(args.log, level=args.log_level, append=args.log_append)
    log.debug(f"command line: {' '.join(sys.argv)}")

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        log.error("Interrupted by user")
        return 130
    except Exception as exc:
        log.error(f"{type(exc).__name__}: {exc}")
        log.debug(traceback.format_exc())
        log.error(f"Full traceback written to {args.log}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
