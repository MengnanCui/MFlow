#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFlow v0.6.0 — single-file MACE toolkit.

    calc  : evaluate an xyz dataset with MACE -> energies & forces
    relax : optimise every structure with an ASE optimiser -> geometries, energies, steps
    plot  : re-analyse / re-plot an already evaluated file

Writes  <stem>_mace.xyz / <stem>_relax.xyz (labels <prefix>energy, <prefix>forces,
        and for relax <prefix>steps / <prefix>converged / <prefix>Erelax),
        a summary png, a metrics json and py.log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:                                   # peak memory reporting (unix only)
    from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage
except ImportError:                    # pragma: no cover
    getrusage = lambda _: type("rusage", (), {"ru_maxrss": 0})()
    RUSAGE_SELF = RUSAGE_CHILDREN = 0

import numpy as np
import ase.io
from ase import Atoms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, ProgressColumn,
                          TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

__version__ = "0.6.0"

# --- plot style (publication) -------------------------------------------------
COLORS = ["#2470a0", "#ca3e47", "#f29c2b", "#1f640a", "#2ca02c", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#005555"]
MARKERS, LINESTYLES = ["o", "v", "s", "*", "p", "P"], ["-", "--", "-.", ":"]
plt.rcParams.update({
    "font.size": 15, "axes.linewidth": 2, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 2, "ytick.major.width": 2, "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
    "lines.linewidth": 2, "lines.markersize": 13, "lines.markeredgewidth": 2,
    "legend.fontsize": 10, "legend.markerscale": 1.5, "legend.framealpha": 0.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

DEFAULT_MODEL = "/home/cmn01/software/mace-main032026/models/mace-mpa-0-medium.model"
MODEL_ALIASES = {"mp": "medium", "mp0": "medium", "mpa": "medium-mpa-0", "mpa0": "medium-mpa-0",
                 "omat": "medium-omat-0", "omat0": "medium-omat-0"}
REF_CANDIDATES = ["", "REF_", "ref_", "dft_", "DFT_", "pbe_", "scan_"]
HEXBIN_ABOVE, PLOT_MAX_POINTS = 20_000, 500_000
GIB = 1024 ** 3   # memory is reported in GiB, the unit nvidia-smi and free agree on
# shapes that could be mistaken for a per-atom array in a small cell
AMBIGUOUS = {"stress", "dipole", "magmom"}

EXAMPLES = """examples
  python mflow.py calc -in data.xyz                        # default model, batch 32
  python mflow.py calc -in data.xyz -model mpa -batch 64   # download a foundation model
  python mflow.py calc -in data.xyz -model ./my.model -prefix mace_
  python mflow.py calc -in data.xyz -ref dft_              # compare with dft_energy/dft_forces
  python mflow.py calc -in data.xyz -group config_type     # errors split by label
  python mflow.py plot -in data_mace.xyz -group formula    # re-plot, no recompute

  python mflow.py relax -in data.xyz                        # relax positions + cell, fmax 0.05
  python mflow.py relax -in data.xyz -cell none -fmax 0.02  # fixed cell, tighter
  python mflow.py relax -in data.xyz -nproc 3 -dtype float32    # 3 workers sharing one gpu
  python mflow.py relax -in data.xyz -resume                # continue an interrupted run"""


# ============================================================================ #
#  output : one call prints coloured to the terminal and plain to the log file
# ============================================================================ #

class Log:
    """Terminal (colour) + log file (plain) in one call. Time stamps only on rules."""

    THEME = Theme({"key": "cyan", "ok": "green", "warn": "yellow", "err": "bold red", "hi": "bold"})

    def __init__(self):
        self.term = Console(theme=self.THEME, highlight=False)
        self.file = self.fh = None

    def open(self, path: str):
        self.fh = open(path, "w", encoding="utf-8")
        self.file = Console(file=self.fh, theme=self.THEME, no_color=True, width=110, highlight=False)

    def __call__(self, *args, **kwargs):
        self.term.print(*args, **kwargs)
        if self.file:
            self.file.print(*args, **kwargs)
            self.fh.flush()

    def rule(self, text: str):
        """Section marker — this is where the time stamp goes."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self.term.rule(f"[dim]{stamp}[/dim] [hi]{text}[/hi]", align="left", style="cyan")
        if self.file:
            self.file.rule(f"{stamp} {text}", align="left")
            self.fh.flush()

    def kv(self, items: dict, cols: int = 2):
        """Dense key/value block: `cols` key-value pairs per row, no borders."""
        table = Table.grid(padding=(0, 2))
        for _ in range(cols):
            table.add_column(style="key", justify="right")
            table.add_column(max_width=46)
        pairs = [(str(k), str(v)) for k, v in items.items()]
        pairs += [("", "")] * (-len(pairs) % cols)
        for i in range(0, len(pairs), cols):
            table.add_row(*[x for pair in pairs[i:i + cols] for x in pair])
        self(table)


LOG = Log()


def table(title: str, columns: list, rows: list, styles: list | None = None) -> Table:
    t = Table(title=title, title_justify="left", title_style="bold", header_style="bold cyan",
              show_edge=False, box=None, pad_edge=False, padding=(0, 2))
    styles = list(styles or []) + [None] * len(columns)
    for i, col in enumerate(columns):
        t.add_column(col, justify="left" if i == 0 else "right", style=styles[i], no_wrap=True)
    for row in rows:
        t.add_row(*[str(x) for x in row])
    return t


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def num(value: float, digits: int = 2) -> str:
    """Fixed point while it stays readable, scientific once it would blow up the column."""
    return f"{value:.{digits}f}" if abs(value) < 1e5 else f"{value:.3g}"


# ============================================================================ #
#  resource monitor — what the run is actually using, live on the progress bar
# ============================================================================ #

class Monitor:
    """
    GPU / CPU / RAM sampling with a fallback chain, so a missing package never
    costs more than the reading itself. Numbers are whole-device: with several
    workers on one card that is exactly what you need to size -nproc, but it is
    not "how much MFlow uses".
    """

    def __init__(self):
        self.gpu, self.cpu, self.ram = None, None, None   # backend names, None = unavailable
        self.history = {"gpu_util": [], "gpu_mem": [], "cpu": [], "ram": []}
        self.gpu_total = self.ram_total = 0.0             # GB
        self._nvml = self._psutil = None
        self._cpu_prev = None
        if not os.environ.get("MFLOW_NO_MONITOR"):        # escape hatch if a backend misbehaves
            self._setup()

    # -- backends ----------------------------------------------------------- #
    def _setup(self):
        try:                                              # nvml needs no CUDA context
            import pynvml
            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            self.gpu = "pynvml"
            self.gpu_total = pynvml.nvmlDeviceGetMemoryInfo(self._nvml[1]).total / GIB
        except Exception:
            if shutil.which("nvidia-smi"):
                self.gpu = "nvidia-smi"
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(interval=None)             # prime the delta
            self.cpu = self.ram = "psutil"
            self.ram_total = psutil.virtual_memory().total / GIB
        except Exception:
            if Path("/proc/stat").exists():
                self.cpu, self._cpu_prev = "/proc/stat", self._proc_cpu_raw()
            if Path("/proc/meminfo").exists():
                self.ram = "/proc/meminfo"

    @staticmethod
    def _proc_cpu_raw():
        fields = [float(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        return sum(fields), sum(fields) - fields[3] - (fields[4] if len(fields) > 4 else 0.0)

    def _read_gpu(self) -> tuple:
        if self.gpu == "pynvml":
            pynvml, handle = self._nvml
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu), memory.used / GIB
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        util, used, total = (float(x) for x in out.splitlines()[0].split(","))   # MiB
        self.gpu_total = total / 1024
        return util, used / 1024

    def _read_cpu(self) -> float:
        if self.cpu == "psutil":
            return self._psutil.cpu_percent(interval=None) * (os.cpu_count() or 1) / 100
        total, busy = self._proc_cpu_raw()
        prev_total, prev_busy = self._cpu_prev
        self._cpu_prev = (total, busy)
        span = total - prev_total
        return (busy - prev_busy) / span * (os.cpu_count() or 1) if span > 0 else 0.0

    def _read_ram(self) -> float:
        if self.ram == "psutil":
            memory = self._psutil.virtual_memory()
            return (memory.total - memory.available) / GIB
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines()[:5]:
            key, _, value = line.partition(":")
            info[key] = float(value.split()[0]) / 1024 ** 2                # kB -> GiB
        self.ram_total = info.get("MemTotal", 0.0)
        return info.get("MemTotal", 0.0) - info.get("MemAvailable", 0.0)

    # -- sampling ----------------------------------------------------------- #
    def sample(self) -> dict:
        """One reading. Any backend that throws is switched off for the rest of the run."""
        now = {}
        for name, reader, keys_ in (("gpu", self._read_gpu, ("gpu_util", "gpu_mem")),
                                    ("cpu", self._read_cpu, ("cpu",)),
                                    ("ram", self._read_ram, ("ram",))):
            if getattr(self, name) is None:
                continue
            try:
                values = reader()
                values = values if isinstance(values, tuple) else (values,)
                for key, value in zip(keys_, values):
                    now[key], _ = value, self.history[key].append(value)
            except Exception:
                setattr(self, name, None)                 # never let monitoring break a run
        return now

    # -- rendering ---------------------------------------------------------- #
    @staticmethod
    def _band(fraction: float, low_is_bad: bool = True) -> str:
        """Idle GPU means raise -nproc; nearly full memory means lower it."""
        if low_is_bad:
            return "ok" if fraction >= 0.7 else ("warn" if fraction >= 0.3 else "err")
        return "err" if fraction >= 0.9 else ("warn" if fraction >= 0.75 else "ok")

    def line(self) -> str:
        """
        Compact markup for the progress bar. Parts are dropped from the right when the
        terminal is too narrow, so GPU — the number you tune -nproc by — survives longest.
        """
        now, parts = self.sample(), []                    # (plain text, markup)
        if "gpu_util" in now:
            util, mem = now["gpu_util"], now.get("gpu_mem", 0.0)
            plain = f"GPU {util:3.0f}% {mem:.1f}/{self.gpu_total:.0f}GB"
            parts.append((plain, f"[{self._band(util / 100)}]GPU {util:3.0f}%[/] "
                                 f"[{self._band(mem / self.gpu_total if self.gpu_total else 0, False)}]"
                                 f"{mem:.1f}/{self.gpu_total:.0f}GB[/]"))
        if "cpu" in now:
            cores = os.cpu_count() or 1
            style = "warn" if now["cpu"] > 0.95 * cores else "dim"   # busy is fine, oversubscribed is not
            plain = f"CPU {now['cpu'] * 100:.0f}%"
            parts.append((plain, f"[{style}]{plain}[/]"))
        if "ram" in now:
            plain = f"RAM {now['ram']:.1f}/{self.ram_total:.0f}GB"
            parts.append((plain, f"[{self._band(now['ram'] / self.ram_total if self.ram_total else 0, False)}]"
                                 f"{plain}[/]"))

        budget = LOG.term.width - 60                      # what the bar itself needs
        while parts and 2 + sum(len(p) for p, _ in parts) + 3 * (len(parts) - 1) > budget:
            parts.pop()
        return "[dim]│[/dim] " + " [dim]·[/dim] ".join(m for _, m in parts) if parts else ""

    def summary(self) -> dict:
        """min / mean / max per metric, no time series kept."""
        out = {}
        for key, values in self.history.items():
            if values:
                out[key] = {"mean": round(float(np.mean(values)), 2),
                            "peak": round(float(np.max(values)), 2)}
        out["backends"] = {"gpu": self.gpu, "cpu": self.cpu, "ram": self.ram}
        if self.gpu_total:
            out["gpu_total_gb"] = round(self.gpu_total, 1)
        if self.ram_total:
            out["ram_total_gb"] = round(self.ram_total, 1)
        return out

    def report(self):
        data, parts = self.summary(), []
        if "gpu_util" in data:
            parts.append(f"GPU util mean {data['gpu_util']['mean']:.0f}% peak {data['gpu_util']['peak']:.0f}%")
        if "gpu_mem" in data:
            parts.append(f"GPU mem peak {data['gpu_mem']['peak']:.1f}/{self.gpu_total:.1f} GB")
        if "cpu" in data:
            parts.append(f"CPU mean {data['cpu']['mean'] * 100:.0f}% of {(os.cpu_count() or 1) * 100}%")
        if "ram" in data:
            parts.append(f"RAM peak {data['ram']['peak']:.1f}/{self.ram_total:.1f} GB")
        if parts:
            LOG(f"[dim]resources ·[/dim] " + " [dim]·[/dim] ".join(parts))


MONITOR = Monitor()


class ResourceColumn(ProgressColumn):
    """Live resource readout appended to the bar. rich re-renders it at most once a second."""

    max_refresh = 1.0

    def render(self, task) -> Text:
        return Text.from_markup(MONITOR.line(), style="none")


def progress_columns(unit: str) -> tuple:
    return ("[progress.description]{task.description}", BarColumn(), MofNCompleteColumn(), unit,
            TimeElapsedColumn(), "eta", TimeRemainingColumn(), ResourceColumn())


# ============================================================================ #
#  data
# ============================================================================ #

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


# ============================================================================ #
#  MACE
# ============================================================================ #

def resolve_model(spec: str) -> tuple[str, str]:
    """-> ('file', path) for a local model, else ('foundation', name to download)."""
    path = os.path.expanduser(spec or "")
    if os.path.isfile(path):
        return "file", path
    if "/" in path or path.endswith(".model"):
        sys.exit(f"model file not found: {path}")
    return "foundation", MODEL_ALIASES.get(spec.lower(), spec)


def load_model(spec: str, device: str, dtype: str):
    """Raw torch module + device, from a local file or a foundation model name."""
    import torch
    from mace.tools import torch_tools

    torch_tools.set_default_dtype(dtype)
    dev = torch_tools.init_device(device)
    kind, value = resolve_model(spec)
    if kind == "file":
        try:
            model = torch.load(value, map_location=dev, weights_only=False)
        except TypeError:
            model = torch.load(value, map_location=dev)
    else:
        from mace.calculators import mace_mp
        model = mace_mp(model=value, device=str(dev), default_dtype=dtype, return_raw_model=True)

    model = (model.double() if dtype == "float64" else model.float()).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    LOG(f"model [key]{value}[/key] · {sum(p.numel() for p in model.parameters()):,} params · "
        f"r_max {float(model.r_max):.1f} Å · {dev} · {dtype}")
    return model, dev


def evaluate(frames: list, spec: str, batch_size: int, device: str, dtype: str,
             prefix: str, head: str | None) -> list:
    """Batched forward pass; results go into info[<prefix>energy] / arrays[<prefix>forces]."""
    import torch
    import mace.data as mdata
    from mace.tools import torch_geometric, torch_tools, utils

    model, dev = load_model(spec, device, dtype)
    ekey, fkey = keys(prefix)

    def bare(atoms):  # geometry only: keeps MACE away from the reference labels
        new = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        if head:
            new.info["head"] = head
        return new

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", None)
    dataset = [mdata.AtomicData.from_config(mdata.config_from_atoms(bare(a)), z_table=z_table,
                                            cutoff=float(model.r_max), heads=heads) for a in frames]
    loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    energy_chunks, force_chunks, t0 = [], [], time.perf_counter()
    with Progress(*progress_columns("batches"), console=LOG.term, transient=True) as progress:
        task = progress.add_task("evaluating", total=len(loader))
        for batch in loader:
            batch = batch.to(dev)
            out = model(batch.to_dict(), compute_stress=False)
            energy_chunks.append(torch_tools.to_numpy(out["energy"]))
            force_chunks += np.split(torch_tools.to_numpy(out["forces"]),
                                     batch.ptr[1:].cpu().numpy(), axis=0)[:-1]
            progress.advance(task)

    for atoms, energy, force in zip(frames, np.concatenate(energy_chunks), force_chunks):
        atoms.calc, atoms.info[ekey], atoms.arrays[fkey] = None, float(energy), np.asarray(force)

    walltime = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    LOG(f"[ok]done[/ok] {len(frames)} structures in {elapsed(walltime)} "
        f"· {len(frames) / walltime:.1f} struct/s · {sum(map(len, frames)) / walltime:.0f} atom/s "
        f"· batch {batch_size}")
    return frames


def evaluate_ase(frames: list, spec: str, device: str, dtype: str, prefix: str, head: str | None) -> list:
    """Fallback: one structure at a time through the ASE calculator."""
    kind, value = resolve_model(spec)
    kwargs = {"device": device, "default_dtype": dtype, **({"head": head} if head else {})}
    if kind == "file":
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=value, **kwargs)
    else:
        from mace.calculators import mace_mp
        calc = mace_mp(model=value, **kwargs)

    ekey, fkey = keys(prefix)
    t0 = time.perf_counter()
    with Progress(*progress_columns("structures"), console=LOG.term, transient=True) as progress:
        task = progress.add_task("evaluating", total=len(frames))
        for atoms in frames:
            work = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            work.calc = calc
            atoms.calc = None
            atoms.info[ekey] = float(work.get_potential_energy())
            atoms.arrays[fkey] = work.get_forces()
            progress.advance(task)
    LOG(f"[ok]done[/ok] {len(frames)} structures in {elapsed(time.perf_counter() - t0)} (ASE engine)")
    return frames


def run_mace(frames: list, **kwargs) -> list:
    """Batched by default; any failure (API drift, OOM) drops to the ASE calculator."""
    try:
        return evaluate(frames, **kwargs)
    except Exception as exc:
        LOG(f"[warn]batched run failed ({type(exc).__name__}: {exc}) — retrying one by one[/warn]")
        kwargs.pop("batch_size")
        return evaluate_ase(frames, **kwargs)


# ============================================================================ #
#  metrics
# ============================================================================ #

def stats(pred: np.ndarray, ref: np.ndarray, scale: float = 1.0) -> dict:
    """MAE / RMSE / max|Δ| (in `scale` units) and R² for one pair of arrays."""
    d = (np.asarray(pred) - np.asarray(ref)) * scale
    r2 = float(np.corrcoef(ref, pred)[0, 1] ** 2) if len(ref) > 1 and np.ptp(ref) else float("nan")
    return {"MAE": float(np.abs(d).mean()), "RMSE": float(np.sqrt((d ** 2).mean())),
            "max": float(np.abs(d).max()), "R2": r2, "N": int(d.size)}


def composition(frames: list) -> tuple[np.ndarray, list]:
    """C[i, j] = how many atoms of element j structure i contains."""
    zs = sorted({int(z) for a in frames for z in a.numbers})
    return np.array([[int(np.count_nonzero(a.numbers == z)) for z in zs] for a in frames], float), zs


def align_e0(frames: list, diff: np.ndarray, mode) -> tuple[np.ndarray, np.ndarray, list]:
    """
    No two codes share an atomic reference, so total energies are only comparable up to
    one constant per element:  E_pred - E_ref = Σ_j n_j δ_j.  This returns δ (eV/atom of
    that element), fitted from the dataset itself — no isolated-atom energies needed.

        fit  : least squares over the composition matrix (default; needs independent compositions)
        mean : a single δ shared by all elements — what a plain mean shift does
        none : δ = 0, compare the raw numbers
        dict : δ supplied by the user, {"Si": -0.12, ...} or {"14": -0.12, ...}
    """
    from ase.data import chemical_symbols
    C, zs = composition(frames)
    if mode == "none":
        return np.zeros(len(zs)), C, zs
    if isinstance(mode, dict):
        delta = np.array([float(mode.get(chemical_symbols[z], mode.get(str(z), 0.0))) for z in zs])
    elif mode == "fit":
        delta, _, rank, _ = np.linalg.lstsq(C, diff, rcond=None)  # minimum-norm solution
        if rank < len(zs):
            LOG(f"[warn]E0 fit: only {rank} of {len(zs)} element shifts are identifiable — the total "
                f"correction is still exact, but the per-element numbers below are one of many solutions[/warn]")
        if len(frames) < 3 * len(zs):
            LOG(f"[warn]E0 fit on {len(frames)} structures for {len(zs)} elements — the fit can absorb "
                f"real error; check against '-e0 none'[/warn]")
    else:
        delta = np.full(len(zs), float((diff / C.sum(1)).mean()))
    return delta, C, zs


def compare(frames: list, pred: str, ref: str, e0="fit") -> tuple[dict, dict]:
    """Errors (meV/atom, meV/Å) after E0 alignment + the arrays the figure needs."""
    from ase.data import chemical_symbols
    n = np.array([len(a) for a in frames], dtype=float)
    et_pred, et_ref = energies(frames, pred), energies(frames, ref)  # total energies, eV
    delta, C, zs = align_e0(frames, et_pred - et_ref, e0)

    series = {"e_pred": (et_pred - C @ delta) / n, "e_ref": et_ref / n, "n_atoms": n}
    series["dE"] = (series["e_pred"] - series["e_ref"]) * 1000            # meV/atom, signed
    errors = {"energy": stats(series["e_pred"], series["e_ref"], 1000),
              "energy_raw": stats(et_pred / n, et_ref / n, 1000),
              "e0_mode": e0 if isinstance(e0, str) else "user",
              "e0_shift_eV": {chemical_symbols[z]: float(d) for z, d in zip(zs, delta)}}
    if keys(ref)[1] in frames[0].arrays:
        series["f_pred"], series["f_ref"] = forces(frames, pred), forces(frames, ref)
        errors["forces"] = stats(series["f_pred"], series["f_ref"], 1000)  # forces need no alignment
        fp, fr = keys(pred)[1], keys(ref)[1]
        sq = np.array([np.sum((a.arrays[fp] - a.arrays[fr]) ** 2) for a in frames])  # eV²/Å²
        series["dF"] = np.sqrt(sq / (3 * n)) * 1000                        # meV/Å, per structure
        series["sqF"] = sq
    return errors, series


def report_errors(errors: dict, pred: str, ref: str):
    rows = [(name, num(s["MAE"]), num(s["RMSE"]), num(s["max"]), f"{s['R2']:.4f}", f"{s['N']:,}", unit)
            for name, s, unit in [("energy (E0 aligned)", errors["energy"], "meV/atom"),
                                  ("energy (raw)", errors["energy_raw"], "meV/atom")]
            + ([("forces", errors["forces"], "meV/Å")] if "forces" in errors else [])]
    LOG(table(f"error  {pred or 'pred'} vs {ref or 'ref'}",
              ["", "MAE", "RMSE", "max|Δ|", "R²", "N", "unit"], rows,
              styles=["key", None, "hi", None, None, "dim", "dim"]))
    shifts = " · ".join(f"{el} {d:+.4f}" for el, d in errors["e0_shift_eV"].items())
    LOG(f"[dim]E0 alignment ({errors['e0_mode']}), eV per atom of element: {shifts}[/dim]")


def report_prediction(frames: list, prefix: str) -> dict:
    n = np.array([len(a) for a in frames], dtype=float)
    e_pa = energies(frames, prefix) / n
    fmax = np.array([np.linalg.norm(a.arrays[keys(prefix)[1]], axis=1).max() for a in frames])
    rows = [("energy", "eV/atom", e_pa), ("max |F|", "eV/Å", fmax)]
    LOG(table(f"prediction  {prefix or ''}", ["", "unit", "min", "max", "mean", "std"],
              [(name, unit, num(v.min(), 4), num(v.max(), 4), num(v.mean(), 4), num(v.std(), 4))
               for name, unit, v in rows], styles=["key", "dim"]))
    return {name: {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                   "std": float(v.std()), "unit": unit} for name, unit, v in rows}


# ============================================================================ #
#  grouping — split the errors by whatever label the dataset already carries
# ============================================================================ #

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


def report_groups(groups: np.ndarray, series: dict, key: str) -> dict:
    """One row per group, worst energy RMSE first — the fast way to see which family fails."""
    out = {}
    for name in dict.fromkeys(groups):
        m = groups == name
        row = {"N": int(m.sum()),
               "E_MAE": float(np.abs(series["dE"][m]).mean()),
               "E_RMSE": float(np.sqrt(np.mean(series["dE"][m] ** 2)))}
        if "sqF" in series:
            row["F_RMSE"] = float(np.sqrt(series["sqF"][m].sum() / (3 * series["n_atoms"][m]).sum()) * 1000)
        out[name] = row

    order = sorted(out, key=lambda k: -out[k]["E_RMSE"])
    has_f = "sqF" in series
    LOG(table(f"by {key}  (worst first)",
              ["group", "N", "E MAE", "E RMSE", "F RMSE"] if has_f else ["group", "N", "E MAE", "E RMSE"],
              [[g, f"{out[g]['N']:,}", num(out[g]["E_MAE"]), num(out[g]["E_RMSE"])]
               + ([num(out[g]["F_RMSE"])] if has_f else []) for g in order],
              styles=["key"]))
    LOG("[dim]E in meV/atom, F in meV/Å · the E0 shift is fitted once on the whole dataset, "
        "so the groups stay comparable[/dim]")
    return out


def report_worst(frames: list, series: dict, groups: np.ndarray | None, top: int = 5):
    """The structures to look at first, by |ΔE| per atom."""
    order = np.argsort(-np.abs(series["dE"]))[:top]
    has_f = "dF" in series
    LOG(table(f"worst {len(order)} structures by |ΔE|",
              ["#", "group", "formula", "atoms", "ΔE", "F RMSE"] if has_f else
              ["#", "group", "formula", "atoms", "ΔE"],
              [[i, groups[i] if groups is not None else "—",
                frames[i].get_chemical_formula(mode="hill", empirical=True), len(frames[i]),
                f"{series['dE'][i]:+.2f}"] + ([num(series["dF"][i])] if has_f else [])
               for i in order], styles=["dim", "key"]))
    LOG("[dim]index is the position in the input file · ΔE in meV/atom, F RMSE in meV/Å[/dim]")


# ============================================================================ #
#  plots — one figure, everything on it
# ============================================================================ #

def _thin(*arrays):
    if arrays[0].size <= PLOT_MAX_POINTS:
        return arrays
    idx = np.random.default_rng(8465).choice(arrays[0].size, PLOT_MAX_POINTS, replace=False)
    LOG(f"[dim]plot: random subsample {PLOT_MAX_POINTS:,} / {arrays[0].size:,} points[/dim]")
    return [a[idx] for a in arrays]


def _parity(ax, x, y, xlabel, ylabel, title, note, groups=None):
    x, y, groups = _thin(np.asarray(x, float), np.asarray(y, float),
                         np.asarray(groups) if groups is not None else np.zeros(len(x)))
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.05 or 1.0
    names = sorted(set(groups.tolist()))
    size = 80 if x.size <= 500 else (40 if x.size <= 3000 else 12)  # readable when points overlap
    if 1 < len(names) <= 12:
        for i, name in enumerate(names):
            m = groups == name
            ax.scatter(x[m], y[m], color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                       alpha=0.7, s=size, edgecolors=COLORS[i % len(COLORS)], label=str(name),
                       linewidths=1.5 if size >= 40 else 0.5, zorder=3)
        ax.legend(framealpha=0.5, fontsize=9, loc="lower right", markerscale=1.2)
    elif x.size > HEXBIN_ABOVE:
        from matplotlib.colors import LogNorm
        hb = ax.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1, norm=LogNorm())
        ax.figure.colorbar(hb, ax=ax, label="counts")
    else:
        ax.scatter(x, y, color=COLORS[0], marker=MARKERS[0], alpha=0.7, s=size,
                   edgecolors=COLORS[0], linewidths=1.5 if size >= 40 else 0.5, zorder=3)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k", ls=LINESTYLES[1], lw=2, zorder=4)
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel=xlabel, ylabel=ylabel, title=title)
    ax.text(0.04, 0.96, note, transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6, ec="none"))


def _bars(ax, names: list, values: list, xlabel: str, color: str):
    """Horizontal so the group names stay readable; worst group on top."""
    order = np.argsort(values)
    ax.barh([str(names[i]) for i in order], [values[i] for i in order], color=color,
            alpha=0.8, edgecolor="black", linewidth=1.0, zorder=2)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=11)


def _hist(ax, values, xlabel, color, zero_line=True):
    ax.hist(_thin(np.asarray(values))[0], bins=50, color=color, alpha=0.75,
            edgecolor="black", linewidth=1.0, zorder=2)
    if zero_line:
        ax.axvline(0, color="black", ls=LINESTYLES[1], lw=2, zorder=3)
    ax.set(xlabel=xlabel, ylabel="Counts")


def make_plot(frames: list, pred: str, ref: str | None, errors: dict | None,
              series: dict | None, outfile: str, groups=None, by_group: dict | None = None):
    """
    No reference        : 1x2 prediction distributions.
    Reference           : 2x2 parity + error histograms.
    Reference + -group  : 2x3, the extra column ranks the groups by RMSE.
    """
    tag = lambda p: (p or "pred").rstrip("_").upper()

    if ref is None or errors is None:
        n = np.array([len(a) for a in frames], dtype=float)
        fmax = np.array([np.linalg.norm(a.arrays[keys(pred)[1]], axis=1).max() for a in frames])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _hist(axes[0], energies(frames, pred) / n, "Energy (eV/atom)", COLORS[0], zero_line=False)
        _hist(axes[1], fmax, r"max |F| (eV/$\mathrm{\AA}$)", COLORS[1], zero_line=False)
    else:
        # energies here are already E0-aligned, so the axes are directly comparable
        e_pred, e_ref, es = series["e_pred"], series["e_ref"], errors["energy"]
        ncols = 3 if by_group else 2
        fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols + 0.5, 8.5))
        _parity(axes[0, 0], e_ref, e_pred, f"{tag(ref)} energy (eV/atom)",
                f"{tag(pred)} energy (eV/atom)", f"Energy · E0 {errors['e0_mode']}",
                f"RMSE = {es['RMSE']:.2f} meV/atom\nR$^2$ = {es['R2']:.4f}\nN = {es['N']:,}", groups)
        _hist(axes[1, 0], series["dE"], "Energy error (meV/atom)", COLORS[0])
        if "forces" in errors:
            f_pred, f_ref, fs = series["f_pred"], series["f_ref"], errors["forces"]
            # one group label per force component
            per_comp = None if groups is None else np.repeat(groups, (3 * series["n_atoms"]).astype(int))
            _parity(axes[0, 1], f_ref, f_pred, rf"{tag(ref)} force comp. (eV/$\mathrm{{\AA}}$)",
                    rf"{tag(pred)} force comp. (eV/$\mathrm{{\AA}}$)", "Forces",
                    f"RMSE = {fs['RMSE']:.1f} meV/$\\mathrm{{\\AA}}$\nR$^2$ = {fs['R2']:.4f}\nN = {fs['N']:,}",
                    per_comp)
            _hist(axes[1, 1], (f_pred - f_ref) * 1000, r"Force error (meV/$\mathrm{\AA}$)", COLORS[1])
        else:
            axes[0, 1].axis("off"), axes[1, 1].axis("off")

        if by_group:
            names = list(by_group)
            _bars(axes[0, 2], names, [by_group[g]["E_RMSE"] for g in names],
                  "Energy RMSE (meV/atom)", COLORS[0])
            if "F_RMSE" in by_group[names[0]]:
                _bars(axes[1, 2], names, [by_group[g]["F_RMSE"] for g in names],
                      r"Force RMSE (meV/$\mathrm{\AA}$)", COLORS[1])
            else:
                axes[1, 2].axis("off")

    for ax in np.atleast_1d(axes).ravel():
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


# ============================================================================ #
#  relax — ASE optimisers, one structure per worker process
# ============================================================================ #

OPTIMIZERS = {"fire": "FIRE", "fire2": "FIRE2", "lbfgs": "LBFGS", "bfgs": "BFGS"}

_W: dict = {}  # per-worker state: the calculator is built once and reused


def build_calculator(spec: str, device: str, dtype: str, head: str | None):
    """ASE calculator for a local .model file or a foundation model name."""
    kind, value = resolve_model(spec)
    kwargs = {"device": device, "default_dtype": dtype, **({"head": head} if head else {})}
    if kind == "file":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=value, **kwargs)
    from mace.calculators import mace_mp
    return mace_mp(model=value, **kwargs)


def _relax_init(spec, device, dtype, head, threads, settings):
    """Runs once per worker process — loading the model per structure would dominate."""
    import torch
    torch.set_num_threads(max(1, threads))
    _W.update(settings)
    _W["calc"] = build_calculator(spec, device, dtype, head)


def _relax_one(payload):
    """One structure, start to finish. Never raises: failures come back as a message."""
    index, numbers, positions, cell, pbc = payload
    t0 = time.perf_counter()
    result = {"index": index, "error": None, "cell_relaxed": False, "gpu_mb": 0.0}
    try:
        from ase.optimize import BFGS, FIRE, FIRE2, LBFGS
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        atoms.calc = _W["calc"]
        energy0 = float(atoms.get_potential_energy())
        volume0 = atoms.get_volume() if bool(np.any(pbc)) else 0.0

        # a zero-volume / non-periodic cell has no strain to relax against
        target, relax_cell = atoms, _W["cell"] == "full" and volume0 > 1e-6
        if relax_cell:
            from ase.filters import FrechetCellFilter
            target = FrechetCellFilter(atoms)

        optimiser = {"FIRE": FIRE, "FIRE2": FIRE2, "LBFGS": LBFGS, "BFGS": BFGS}[_W["opt"]]
        opt = optimiser(target, logfile=None)

        traj = []
        if _W["traj"]:
            opt.attach(lambda: traj.append((atoms.get_positions().copy(),
                                            np.array(atoms.cell), float(atoms.get_potential_energy()))))
        converged = bool(opt.run(fmax=_W["fmax"], steps=_W["steps"]))

        forces = atoms.get_forces()
        result.update(
            positions=atoms.get_positions(), cell=np.array(atoms.cell),
            energy=float(atoms.get_potential_energy()), energy0=energy0, forces=forces,
            steps=int(opt.get_number_of_steps()), converged=converged,
            fmax=float(np.linalg.norm(forces, axis=1).max()),
            dmax=float(np.linalg.norm(atoms.get_positions() - positions, axis=1).max()),
            dvol=float((atoms.get_volume() - volume0) / volume0 * 100) if relax_cell else 0.0,
            cell_relaxed=relax_cell, traj=traj,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    result["seconds"] = time.perf_counter() - t0
    return result


def apply_result(atoms: Atoms, prefix: str, res: dict):
    """Write one worker result back into the structure, leaving every original label alone."""
    if res["error"]:
        atoms.info[f"{prefix}steps"], atoms.info[f"{prefix}converged"] = 0, False
        atoms.info[f"{prefix}error"] = res["error"]
    else:
        atoms.set_positions(res["positions"])
        if res["cell_relaxed"]:
            atoms.set_cell(res["cell"])
            atoms.info[f"{prefix}dvol"] = round(res["dvol"], 4)
        atoms.info[f"{prefix}energy"] = res["energy"]
        atoms.arrays[f"{prefix}forces"] = res["forces"]
        atoms.info[f"{prefix}steps"] = res["steps"]
        atoms.info[f"{prefix}converged"] = res["converged"]
        atoms.info[f"{prefix}fmax"] = round(res["fmax"], 6)
        atoms.info[f"{prefix}energy0"] = res["energy0"]
        atoms.info[f"{prefix}Erelax"] = (res["energy"] - res["energy0"]) / len(atoms)
        atoms.info[f"{prefix}dmax"] = round(res["dmax"], 4)
    atoms.info[f"{prefix}walltime"] = round(res["seconds"], 2)
    atoms.info[f"{prefix}index"] = res["index"]


def write_traj(handle, atoms: Atoms, prefix: str, res: dict):
    """All optimisation paths go into one file, tagged by structure and step."""
    for step, (positions, cell, energy) in enumerate(res.get("traj") or [], start=1):
        frame = Atoms(numbers=atoms.numbers, positions=positions, cell=cell, pbc=atoms.pbc)
        frame.info.update({f"{prefix}opt_index": res["index"], f"{prefix}opt_step": step,
                           f"{prefix}energy": energy})
        ase.io.write(handle, frame, format="extxyz")


def pick_nproc(value: str, device: str) -> int:
    """One worker per GPU by default; on CPU leave room for each worker's own threads."""
    if value != "auto":
        return max(1, int(value))
    return 1 if device != "cpu" else max(1, (os.cpu_count() or 4) // 4)


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


def load_part(frames: list, part: Path, prefix: str) -> set:
    """Structures already finished in a previous run are restored whole, labels included."""
    if not part.exists():
        return set()
    try:
        done_frames = ase.io.read(part, index=":")
    except Exception as exc:
        LOG(f"[warn]could not read {part} ({type(exc).__name__}) — starting over[/warn]")
        return set()
    done = set()
    for atoms in done_frames:
        index = int(atoms.info.get(f"{prefix}index", -1))
        if 0 <= index < len(frames):
            frames[index], _ = atoms, done.add(index)
    LOG(f"resuming: [hi]{len(done)}[/hi] structures already done in {part}")
    return done


def report_relax(frames: list, prefix: str, done_now: int, wall: float,
                 nproc: int, gpu_mb: float) -> dict:
    """Distributions rather than 1000 log lines."""
    def col(key, default=0.0):
        return np.array([float(a.info.get(f"{prefix}{key}", default)) for a in frames])

    steps, erelax, fmax, dmax = col("steps"), col("Erelax"), col("fmax"), col("dmax")
    ok = np.array([not a.info.get(f"{prefix}error") for a in frames])
    converged = np.array([bool(a.info.get(f"{prefix}converged", False)) for a in frames])

    rows = [("steps", "", steps), ("Erelax", "eV/atom", erelax),
            ("final fmax", "eV/Å", fmax), ("displacement", "Å", dmax)]
    LOG(table(f"relax  {prefix}", ["", "unit", "min", "median", "max", "mean"],
              [(name, unit, num(v.min(), 4), num(np.median(v), 4), num(v.max(), 4), num(v.mean(), 4))
               for name, unit, v in rows], styles=["key", "dim"]))

    rate = done_now / wall if wall else 0.0
    LOG(f"[ok]{converged.sum()}[/ok]/{len(frames)} converged · "
        f"[warn]{(~converged & ok).sum()}[/warn] hit the step limit · "
        f"[err]{(~ok).sum()}[/err] failed · {elapsed(wall)} · {rate:.2f} struct/s · "
        f"{nproc} worker(s)" + (f" · GPU peak {gpu_mb:,.0f} MB/worker" if gpu_mb else ""))

    MONITOR.report()

    stuck = [i for i, a in enumerate(frames) if not a.info.get(f"{prefix}converged", False)][:10]
    if stuck:
        LOG(f"[dim]not converged: index {', '.join(map(str, stuck))}"
            f"{' …' if (~converged).sum() > 10 else ''}[/dim]")
    for i, a in enumerate(frames):
        if a.info.get(f"{prefix}error"):
            LOG(f"[err]failed[/err] index {i}: {a.info[f'{prefix}error']}")

    return {"resources": MONITOR.summary(),
            "converged": int(converged.sum()), "failed": int((~ok).sum()), "n": len(frames),
            "steps": {"min": int(steps.min()), "median": float(np.median(steps)), "max": int(steps.max())},
            "Erelax_eV_per_atom": {"mean": float(erelax.mean()), "min": float(erelax.min())},
            "walltime_s": round(wall, 1), "struct_per_s": round(rate, 3),
            "nproc": nproc, "gpu_peak_mb_per_worker": round(gpu_mb, 1)}


def report_relax_groups(frames: list, prefix: str, groups: np.ndarray, key: str) -> dict:
    """Which family of structures is hard to relax."""
    steps = np.array([float(a.info.get(f"{prefix}steps", 0)) for a in frames])
    erelax = np.array([float(a.info.get(f"{prefix}Erelax", 0.0)) for a in frames])
    converged = np.array([bool(a.info.get(f"{prefix}converged", False)) for a in frames])

    out = {}
    for name in dict.fromkeys(groups):
        m = groups == name
        out[name] = {"N": int(m.sum()), "steps_median": float(np.median(steps[m])),
                     "converged_pct": float(100 * converged[m].mean()),
                     "Erelax_mean": float(erelax[m].mean())}
    order = sorted(out, key=lambda k: -out[k]["steps_median"])
    LOG(table(f"by {key}  (slowest first)", ["group", "N", "steps (median)", "converged", "Erelax mean"],
              [[g, f"{out[g]['N']:,}", num(out[g]["steps_median"], 1),
                f"{out[g]['converged_pct']:.0f} %", num(out[g]["Erelax_mean"], 4)] for g in order],
              styles=["key"]))
    return out


def _scatter(ax, x, y, xlabel, ylabel, groups=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    names = sorted(set(groups.tolist())) if groups is not None else []
    if 1 < len(names) <= 12:
        for i, name in enumerate(names):
            m = groups == name
            ax.scatter(x[m], y[m], color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                       alpha=0.7, s=40, edgecolors=COLORS[i % len(COLORS)], linewidths=1.5,
                       zorder=3, label=str(name))
        ax.legend(framealpha=0.5, fontsize=9, markerscale=1.2)
    else:
        ax.scatter(x, y, color=COLORS[0], marker=MARKERS[0], alpha=0.7, s=40,
                   edgecolors=COLORS[0], linewidths=1.5, zorder=3)
    ax.set(xlabel=xlabel, ylabel=ylabel)


def make_relax_plot(frames: list, prefix: str, fmax_target: float, outfile: str, groups=None):
    steps = np.array([float(a.info.get(f"{prefix}steps", 0)) for a in frames])
    erelax = np.array([float(a.info.get(f"{prefix}Erelax", 0.0)) for a in frames]) * 1000
    fmax = np.array([float(a.info.get(f"{prefix}fmax", 0.0)) for a in frames])
    natoms = np.array([len(a) for a in frames], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    _hist(axes[0, 0], steps, "Optimisation steps", COLORS[0], zero_line=False)
    _hist(axes[0, 1], erelax, "Relaxation energy (meV/atom)", COLORS[1], zero_line=True)
    _hist(axes[1, 0], np.log10(np.clip(fmax, 1e-6, None)),
          r"log$_{10}$ final max|F| (eV/$\mathrm{\AA}$)", COLORS[2], zero_line=False)
    axes[1, 0].axvline(np.log10(fmax_target), color="black", ls=LINESTYLES[1], lw=2, zorder=3)
    _scatter(axes[1, 1], natoms, steps, "Atoms per structure", "Optimisation steps", groups)

    for ax in axes.ravel():
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


def cmd_relax(args) -> int:
    import multiprocessing as mp

    t0 = time.perf_counter()
    stem = Path(args.input).stem
    outfile = args.output or str(Path(args.outdir) / f"{stem}_relax.xyz")
    device, (kind, model) = pick_device(args.device), resolve_model(args.model)
    nproc = pick_nproc(args.nproc, device)
    threads = max(1, (os.cpu_count() or 4) // nproc)

    LOG.rule(f"MFlow {__version__} · relax")
    LOG.kv({"input": args.input, "index": args.index,
            "model": model, "kind": "local file" if kind == "file" else "foundation",
            "optimiser": args.opt.upper(), "cell": args.cell,
            "fmax": f"{args.fmax} eV/Å", "max steps": args.steps,
            "workers": f"{nproc} × {threads} thread(s)", "device": device,
            "dtype": args.dtype, "head": args.head or "—",
            "output": outfile, "log": args.log})
    LOG(f"[dim]{versions()}[/dim]")

    LOG.rule("dataset")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    frames = read_dataset(args.input, args.index)
    prefix = free_prefix(frames, fix_prefix(args.prefix), args.overwrite)

    part = Path(args.outdir) / f"{stem}_relax.part.xyz"
    done = load_part(frames, part, prefix) if args.resume else set()
    if not args.resume and part.exists():
        part.unlink()
    todo = [(i, a.numbers, a.get_positions(), np.array(a.cell), a.pbc)
            for i, a in enumerate(frames) if i not in done]

    LOG.rule("relax")
    settings = {"opt": OPTIMIZERS[args.opt], "cell": args.cell, "fmax": args.fmax,
                "steps": args.steps, "traj": bool(args.traj)}
    initargs = (args.model, device, args.dtype, args.head, threads, settings)

    part_fh = open(part, "a", encoding="utf-8")
    traj_fh = open(Path(args.outdir) / args.traj, "w", encoding="utf-8") if args.traj else None
    gpu_mb, t_run = 0.0, time.perf_counter()
    try:
        with Progress(*progress_columns("structures"), console=LOG.term, transient=True) as progress:
            task = progress.add_task("relaxing", total=len(todo))

            def collect(res):
                nonlocal gpu_mb
                gpu_mb = max(gpu_mb, res.get("gpu_mb", 0.0))
                atoms = frames[res["index"]]
                apply_result(atoms, prefix, res)
                ase.io.write(part_fh, atoms, format="extxyz")
                part_fh.flush()
                if traj_fh:
                    write_traj(traj_fh, atoms, prefix, res)
                progress.advance(task)

            if nproc == 1:                       # no pool: cheaper to start and easier to debug
                _relax_init(*initargs)
                for payload in todo:
                    collect(_relax_one(payload))
            else:
                context = mp.get_context("spawn")   # fork + CUDA does not survive
                with context.Pool(nproc, initializer=_relax_init, initargs=initargs) as pool:
                    for res in pool.imap_unordered(_relax_one, todo, chunksize=1):
                        collect(res)
    finally:
        part_fh.close()
        if traj_fh:
            traj_fh.close()
            LOG(f"trajectory [key]{Path(args.outdir) / args.traj}[/key]")

    LOG.rule("results")
    ase.io.write(outfile, frames, format="extxyz")     # original order, all labels intact
    part.unlink(missing_ok=True)
    LOG(f"xyz [key]{outfile}[/key] · {len(frames)} structures · "
        f"labels {prefix}energy / {prefix}forces / [hi]{prefix}steps[/hi] / {prefix}converged")

    summary = report_relax(frames, prefix, len(todo), time.perf_counter() - t_run, nproc, gpu_mb)
    groups = group_of(frames, args.group)
    if groups is not None:
        summary["by_group"] = report_relax_groups(frames, prefix, groups, args.group)
    if not args.no_plot:
        make_relax_plot(frames, prefix, args.fmax,
                        str(Path(args.outdir) / f"{stem}_relax.png"), groups)

    peak = getrusage(RUSAGE_SELF).ru_maxrss + getrusage(RUSAGE_CHILDREN).ru_maxrss
    summary.update({"mflow": __version__, "input": os.path.abspath(args.input),
                    "output": os.path.abspath(outfile), "model": model, "prefix": prefix,
                    "optimiser": args.opt, "cell": args.cell, "fmax": args.fmax,
                    "max_steps": args.steps, "device": device, "dtype": args.dtype,
                    "peak_rss_gb": round(peak / 1e6, 2), "versions": versions()})
    json_file = Path(args.outdir) / f"{stem}_relax_metrics.json"
    json_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG(f"json [key]{json_file}[/key] · peak RSS {peak / 1e6:.1f} GB · "
        f"total [hi]{elapsed(time.perf_counter() - t0)}[/hi]")
    return 0


# ============================================================================ #
#  commands
# ============================================================================ #

def versions() -> str:
    import platform
    parts = [f"python {platform.python_version()}", f"numpy {np.__version__}"]
    for name in ("ase", "torch", "mace"):
        try:
            parts.append(f"{name} {__import__(name).__version__}")
        except Exception:
            parts.append(f"{name} —")
    try:
        import torch
        parts.append(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
    except Exception:
        pass
    return " · ".join(parts)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def tag_structures(frames: list, prefix: str, series: dict):
    """
    Per-structure diagnostics into info, so the file itself can be sorted, filtered or
    fed back to `-group <prefix>dE`:  dE in meV/atom (signed), dF in meV/Å (RMSE).
    """
    for i, atoms in enumerate(frames):
        atoms.info[f"{prefix}dE"] = float(series["dE"][i])
        if "dF" in series:
            atoms.info[f"{prefix}dF"] = float(series["dF"][i])
    extra = f" / {prefix}dF" if "dF" in series else ""
    LOG(f"[dim]per-structure diagnostics written as {prefix}dE{extra}[/dim]")


def parse_e0(value: str):
    """'fit' | 'mean' | 'none' as is; anything else is a json file of per-element shifts."""
    return value if value in ("fit", "mean", "none") else json.loads(Path(value).read_text(encoding="utf-8"))


def resolve_ref(frames: list, requested: str | None, pred: str) -> str | None:
    if requested is None:
        ref = find_ref(frames, exclude=pred)
        LOG(f"reference [key]{keys(ref)[0]}[/key] (auto)" if ref is not None else
            "[warn]no reference labels found — prediction only[/warn]")
        return ref
    ref = fix_prefix(requested)
    if has_labels(frames[0], ref, forces=False):
        return ref
    LOG(f"[warn]{keys(ref)[0]} not in the dataset — prediction only[/warn]")
    return None


def cmd_calc(args) -> int:
    t0 = time.perf_counter()
    prefix, stem = fix_prefix(args.prefix), Path(args.input).stem
    outfile = args.output or str(Path(args.outdir) / f"{stem}_mace.xyz")
    device, (kind, model) = pick_device(args.device), resolve_model(args.model)

    LOG.rule(f"MFlow {__version__} · calc")
    LOG.kv({"input": args.input, "index": args.index,
            "model": model, "kind": "local file" if kind == "file" else "foundation",
            "batch size": args.batch_size, "device": device,
            "dtype": args.dtype, "head": args.head or "—",
            "labels": " / ".join(keys(prefix)), "reference": args.ref_prefix or "auto",
            "E0 align": args.e0, "group by": args.group or "—",
            "output": outfile, "log": args.log})
    LOG(f"[dim]{versions()}[/dim]")

    LOG.rule("dataset")
    frames = read_dataset(args.input, args.index)
    prefix = free_prefix(frames, prefix, args.overwrite)
    ref = resolve_ref(frames, args.ref_prefix, prefix)

    LOG.rule("mace")
    frames = run_mace(frames, spec=args.model, batch_size=args.batch_size,
                      device=device, dtype=args.dtype, prefix=prefix, head=args.head)

    LOG.rule("results")
    MONITOR.report()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    prediction = report_prediction(frames, prefix)

    errors, series = compare(frames, prefix, ref, parse_e0(args.e0)) if ref is not None else (None, None)
    groups = by_group = None
    if errors:
        report_errors(errors, prefix, ref)
        tag_structures(frames, prefix, series)     # per-structure ΔE / ΔF into the file
        groups = group_of(frames, args.group)
        if groups is not None:
            by_group = report_groups(groups, series, args.group)
        report_worst(frames, series, groups)

    ase.io.write(outfile, frames, format="extxyz")
    LOG(f"xyz [key]{outfile}[/key] · {len(frames)} structures")
    if not args.no_plot:
        make_plot(frames, prefix, ref, errors, series,
                  str(Path(args.outdir) / f"{stem}_summary.png"), groups, by_group)

    metrics = {"mflow": __version__, "time": datetime.now().isoformat(timespec="seconds"),
               "input": os.path.abspath(args.input), "output": os.path.abspath(outfile),
               "model": model, "batch_size": args.batch_size, "device": device, "dtype": args.dtype,
               "prefix": prefix, "ref_prefix": ref, "prediction": prediction, "errors": errors,
               "group_by": args.group, "by_group": by_group, "resources": MONITOR.summary(),
               "versions": versions()}
    json_file = Path(args.outdir) / f"{stem}_metrics.json"
    json_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG(f"json [key]{json_file}[/key] · total [hi]{elapsed(time.perf_counter() - t0)}[/hi]")
    return 0


def cmd_plot(args) -> int:
    pred, stem = fix_prefix(args.pred_prefix), Path(args.input).stem
    LOG.rule(f"MFlow {__version__} · plot")
    frames = read_dataset(args.input, args.index)
    if not has_labels(frames[0], pred, forces=False):
        LOG(f"[err]{keys(pred)[0]} not found in {args.input}[/err]")
        return 1

    ref = resolve_ref(frames, args.ref_prefix, pred)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    errors, series = compare(frames, pred, ref, parse_e0(args.e0)) if ref is not None else (None, None)
    groups = by_group = None
    if errors:
        report_errors(errors, pred, ref)
        groups = group_of(frames, args.group)
        if groups is not None:
            by_group = report_groups(groups, series, args.group)
        report_worst(frames, series, groups)
        (Path(args.outdir) / f"{stem}_metrics.json").write_text(
            json.dumps({"errors": errors, "group_by": args.group, "by_group": by_group}, indent=2),
            encoding="utf-8")
    make_plot(frames, pred, ref, errors, series,
              str(Path(args.outdir) / f"{stem}_summary.png"), groups, by_group)
    return 0


# ============================================================================ #
#  cli
# ============================================================================ #

def build_parser() -> argparse.ArgumentParser:
    """Options are multi-letter on purpose: -in, -out, -model ... no single-letter clashes."""
    parser = argparse.ArgumentParser(prog="mflow.py", epilog=EXAMPLES,
                                     description=f"MFlow {__version__} — MACE energies and forces.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-version", action="version", version=f"MFlow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-in", "--input", dest="input", required=True, metavar="FILE", help="input xyz/extxyz")
    common.add_argument("-index", default=":", help="ase slice, e.g. ':100' [:]")
    common.add_argument("-group", default=None, metavar="KEY",
                        help="split the results by an existing label: any info key "
                             "(config_type, step, ...) or the virtual keys formula / natoms. "
                             "A numeric key is cut into quartiles")
    common.add_argument("-outdir", default=".", help="output directory [.]")
    common.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")

    compare = argparse.ArgumentParser(add_help=False)
    compare.add_argument("-ref", dest="ref_prefix", default=None, metavar="PREFIX",
                         help="reference label prefix, e.g. dft_ [auto-detect]")
    compare.add_argument("-e0", default="fit", metavar="MODE",
                         help="how to align the atomic reference before comparing energies: "
                              "fit (per-element least squares) | mean (one shared shift) | "
                              "none | a json file {'Si': -0.12, ...} [fit]")

    mace_args = argparse.ArgumentParser(add_help=False)
    mace_args.add_argument("-model", default=DEFAULT_MODEL, metavar="PATH|NAME",
                           help="path to a .model file, or a foundation name to download "
                                "(small/medium/large/medium-mpa-0/medium-omat-0, aliases mpa/omat)\n"
                                f"[{DEFAULT_MODEL}]")
    mace_args.add_argument("-prefix", default="mpa0_", help="output label prefix [mpa0_]")
    mace_args.add_argument("-out", dest="output", default=None, metavar="FILE", help="output xyz")
    mace_args.add_argument("-device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="[auto]")
    mace_args.add_argument("-dtype", default="float64", choices=["float64", "float32"], help="[float64]")
    mace_args.add_argument("-head", default=None, help="head of a multi-head model")
    mace_args.add_argument("-overwrite", action="store_true",
                           help="reuse the prefix even when it already exists (default: fall back "
                                "to <prefix>c1_, <prefix>c2_, ... so nothing is overwritten)")
    mace_args.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figure")

    calc = sub.add_parser("calc", parents=[common, compare, mace_args],
                          help="run MACE on an xyz dataset",
                          epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    calc.add_argument("-batch", dest="batch_size", type=int, default=32, help="batch size [32]")
    calc.set_defaults(func=cmd_calc)

    relax = sub.add_parser("relax", parents=[common, mace_args],
                           help="optimise every structure with an ASE optimiser",
                           epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    relax.add_argument("-fmax", type=float, default=0.05,
                       help="convergence on the largest atomic force, eV/A [0.05]")
    relax.add_argument("-steps", type=int, default=500, help="max optimiser steps per structure [500]")
    relax.add_argument("-opt", default="fire", choices=list(OPTIMIZERS), help="ase optimiser [fire]")
    relax.add_argument("-cell", default="full", choices=["full", "none"],
                       help="full = relax the cell too (ase FrechetCellFilter), none = positions "
                            "only [full]. Non-periodic structures always fall back to positions only")
    relax.add_argument("-nproc", default="auto", metavar="N",
                       help="worker processes, each holding its own model "
                            "[auto: 1 on gpu, cores/4 on cpu]")
    relax.add_argument("-traj", default=None, metavar="FILE",
                       help="also write every intermediate frame to this file [off]")
    relax.add_argument("-resume", action="store_true",
                       help="skip structures already finished in <stem>_relax.part.xyz")
    relax.set_defaults(func=cmd_relax)

    plot = sub.add_parser("plot", parents=[common, compare], help="re-analyse an evaluated file")
    plot.add_argument("-pred", dest="pred_prefix", default="mpa0_", metavar="PREFIX",
                      help="prediction prefix [mpa0_]")
    plot.set_defaults(func=cmd_plot)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    LOG.open(args.log)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        LOG("[err]interrupted[/err]")
        return 130
    except Exception:
        LOG.term.print_exception(show_locals=False)
        if LOG.file:
            import traceback
            LOG.file.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
