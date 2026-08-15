#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFlow v0.2.0 — single-file MACE toolkit.

    calc : evaluate an xyz dataset with MACE -> energies & forces
    plot : re-analyse / re-plot an already evaluated file

Writes  <stem>_mace.xyz (labels <prefix>energy / <prefix>forces),
        <stem>_summary.png, <stem>_metrics.json and py.log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.theme import Theme

__version__ = "0.2.0"

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

EXAMPLES = """examples
  python mflow.py calc -in data.xyz                        # default model, batch 32
  python mflow.py calc -in data.xyz -model mpa -batch 64   # download a foundation model
  python mflow.py calc -in data.xyz -model ./my.model -prefix mace_
  python mflow.py calc -in data.xyz -ref dft_              # compare with dft_energy/dft_forces
  python mflow.py plot -in data_mace.xyz -ref dft_         # re-plot, no recompute"""


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
    # extxyz hides plain energy/forces in a SinglePointCalculator — put them back
    for atoms in frames:
        results = getattr(atoms.calc, "results", None) or {}
        if "energy" in results and "energy" not in atoms.info:
            atoms.info["energy"] = float(results["energy"])
        if "forces" in results and "forces" not in atoms.arrays:
            atoms.arrays["forces"] = np.asarray(results["forces"])
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
    columns = ("[progress.description]{task.description}", BarColumn(), MofNCompleteColumn(),
               "batches", TimeElapsedColumn(), "eta", TimeRemainingColumn())
    with Progress(*columns, console=LOG.term, transient=True) as progress:
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
    with Progress(console=LOG.term, transient=True) as progress:
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


def compute_errors(frames: list, pred: str, ref: str) -> dict:
    """Energies per atom (meV/atom, raw and mean-shifted) + force components (meV/Å)."""
    n = np.array([len(a) for a in frames], dtype=float)
    e_pred, e_ref = energies(frames, pred) / n, energies(frames, ref) / n
    out = {"energy": stats(e_pred, e_ref, 1000),
           "energy_shifted": stats(e_pred - e_pred.mean(), e_ref - e_ref.mean(), 1000),
           "offset_meV_per_atom": float((e_pred - e_ref).mean() * 1000)}
    if keys(ref)[1] in frames[0].arrays:
        out["forces"] = stats(forces(frames, pred), forces(frames, ref), 1000)
    return out


def report_errors(errors: dict, pred: str, ref: str):
    rows = [(name, f"{s['MAE']:.2f}", f"{s['RMSE']:.2f}", f"{s['max']:.2f}", f"{s['R2']:.4f}",
             f"{s['N']:,}", unit)
            for name, s, unit in [("energy", errors["energy"], "meV/atom"),
                                  ("energy (shifted)", errors["energy_shifted"], "meV/atom")]
            + ([("forces", errors["forces"], "meV/Å")] if "forces" in errors else [])]
    LOG(table(f"error  {pred or 'pred'} vs {ref or 'ref'}",
              ["", "MAE", "RMSE", "max|Δ|", "R²", "N", "unit"], rows,
              styles=["key", None, "hi", None, None, "dim", "dim"]))
    LOG(f"[dim]constant offset {errors['offset_meV_per_atom']:+.1f} meV/atom "
        f"— the shifted row is the one that describes the shape[/dim]")


def report_prediction(frames: list, prefix: str) -> dict:
    n = np.array([len(a) for a in frames], dtype=float)
    e_pa = energies(frames, prefix) / n
    fmax = np.array([np.linalg.norm(a.arrays[keys(prefix)[1]], axis=1).max() for a in frames])
    rows = [("energy", "eV/atom", e_pa), ("max |F|", "eV/Å", fmax)]
    LOG(table(f"prediction  {prefix or ''}", ["", "unit", "min", "max", "mean", "std"],
              [(name, unit, f"{v.min():.4f}", f"{v.max():.4f}", f"{v.mean():.4f}", f"{v.std():.4f}")
               for name, unit, v in rows], styles=["key", "dim"]))
    return {name: {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                   "std": float(v.std()), "unit": unit} for name, unit, v in rows}


# ============================================================================ #
#  plots — one figure, everything on it
# ============================================================================ #

def _thin(*arrays):
    if arrays[0].size <= PLOT_MAX_POINTS:
        return arrays
    idx = np.random.default_rng(8465).choice(arrays[0].size, PLOT_MAX_POINTS, replace=False)
    LOG(f"[dim]plot: random subsample {PLOT_MAX_POINTS:,} / {arrays[0].size:,} points[/dim]")
    return [a[idx] for a in arrays]


def _parity(ax, x, y, xlabel, ylabel, title, note):
    x, y = _thin(np.asarray(x, float), np.asarray(y, float))
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.05 or 1.0
    if x.size > HEXBIN_ABOVE:
        from matplotlib.colors import LogNorm
        hb = ax.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1, norm=LogNorm())
        ax.figure.colorbar(hb, ax=ax, label="counts")
    else:
        size = 80 if x.size <= 500 else (40 if x.size <= 3000 else 12)
        ax.scatter(x, y, color=COLORS[0], marker=MARKERS[0], alpha=0.7, s=size,
                   edgecolors=COLORS[0], linewidths=1.5 if size >= 40 else 0.5, zorder=3)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k", ls=LINESTYLES[1], lw=2, zorder=4)
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel=xlabel, ylabel=ylabel, title=title)
    ax.text(0.04, 0.96, note, transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6, ec="none"))


def _hist(ax, values, xlabel, color, zero_line=True):
    ax.hist(_thin(np.asarray(values))[0], bins=50, color=color, alpha=0.75,
            edgecolor="black", linewidth=1.0, zorder=2)
    if zero_line:
        ax.axvline(0, color="black", ls=LINESTYLES[1], lw=2, zorder=3)
    ax.set(xlabel=xlabel, ylabel="Counts")


def make_plot(frames: list, pred: str, ref: str | None, errors: dict | None, outfile: str):
    """With a reference: 2x2 parity + error histograms. Without: prediction distributions."""
    n = np.array([len(a) for a in frames], dtype=float)
    e_pred = energies(frames, pred) / n
    tag = lambda p: (p or "pred").rstrip("_").upper()

    if ref is None or errors is None:
        fmax = np.array([np.linalg.norm(a.arrays[keys(pred)[1]], axis=1).max() for a in frames])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _hist(axes[0], e_pred, "Energy (eV/atom)", COLORS[0], zero_line=False)
        _hist(axes[1], fmax, r"max |F| (eV/$\mathrm{\AA}$)", COLORS[1], zero_line=False)
    else:
        e_ref = energies(frames, ref) / n
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        es = errors["energy_shifted"]
        _parity(axes[0, 0], e_ref - e_ref.mean(), e_pred - e_pred.mean(),
                f"{tag(ref)} rel. energy (eV/atom)", f"{tag(pred)} rel. energy (eV/atom)", "Energy",
                f"RMSE = {es['RMSE']:.2f} meV/atom\nR$^2$ = {es['R2']:.4f}\nN = {es['N']:,}")
        _hist(axes[1, 0], (e_pred - e_ref - (e_pred - e_ref).mean()) * 1000,
              "Energy error (meV/atom)", COLORS[0])
        if "forces" in errors:
            f_pred, f_ref, fs = forces(frames, pred), forces(frames, ref), errors["forces"]
            _parity(axes[0, 1], f_ref, f_pred, rf"{tag(ref)} force comp. (eV/$\mathrm{{\AA}}$)",
                    rf"{tag(pred)} force comp. (eV/$\mathrm{{\AA}}$)", "Forces",
                    f"RMSE = {fs['RMSE']:.1f} meV/$\\mathrm{{\\AA}}$\nR$^2$ = {fs['R2']:.4f}\nN = {fs['N']:,}")
            _hist(axes[1, 1], (f_pred - f_ref) * 1000, r"Force error (meV/$\mathrm{\AA}$)", COLORS[1])
        else:
            axes[0, 1].axis("off"), axes[1, 1].axis("off")

    for ax in np.atleast_1d(axes).ravel():
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


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


def resolve_ref(frames: list, requested: str | None, pred: str) -> str | None:
    if requested is None:
        ref = find_ref(frames, exclude=pred)
        LOG(f"reference [key]{keys(ref)[0]}[/key] (auto)" if ref else
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
            "output": outfile, "log": args.log})
    LOG(f"[dim]{versions()}[/dim]")

    LOG.rule("dataset")
    frames = read_dataset(args.input, args.index)
    ref = resolve_ref(frames, args.ref_prefix, prefix)

    LOG.rule("mace")
    frames = run_mace(frames, spec=args.model, batch_size=args.batch_size,
                      device=device, dtype=args.dtype, prefix=prefix, head=args.head)

    LOG.rule("results")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ase.io.write(outfile, frames, format="extxyz")
    LOG(f"xyz [key]{outfile}[/key] · {len(frames)} structures")

    prediction = report_prediction(frames, prefix)
    errors = compute_errors(frames, prefix, ref) if ref else None
    if errors:
        report_errors(errors, prefix, ref)
    if not args.no_plot:
        make_plot(frames, prefix, ref, errors, str(Path(args.outdir) / f"{stem}_summary.png"))

    metrics = {"mflow": __version__, "time": datetime.now().isoformat(timespec="seconds"),
               "input": os.path.abspath(args.input), "output": os.path.abspath(outfile),
               "model": model, "batch_size": args.batch_size, "device": device, "dtype": args.dtype,
               "prefix": prefix, "ref_prefix": ref, "prediction": prediction, "errors": errors,
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
    errors = compute_errors(frames, pred, ref) if ref else None
    if errors:
        report_errors(errors, pred, ref)
        (Path(args.outdir) / f"{stem}_metrics.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    make_plot(frames, pred, ref, errors, str(Path(args.outdir) / f"{stem}_summary.png"))
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
    common.add_argument("-ref", dest="ref_prefix", default=None, metavar="PREFIX",
                        help="reference label prefix, e.g. dft_ [auto-detect]")
    common.add_argument("-outdir", default=".", help="output directory [.]")
    common.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")

    calc = sub.add_parser("calc", parents=[common], help="run MACE on an xyz dataset",
                          epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    calc.add_argument("-model", default=DEFAULT_MODEL, metavar="PATH|NAME",
                      help="path to a .model file, or a foundation name to download "
                           "(small/medium/large/medium-mpa-0/medium-omat-0, aliases mpa/omat)\n"
                           f"[{DEFAULT_MODEL}]")
    calc.add_argument("-batch", dest="batch_size", type=int, default=32, help="batch size [32]")
    calc.add_argument("-prefix", default="mpa0_", help="output label prefix [mpa0_]")
    calc.add_argument("-out", dest="output", default=None, metavar="FILE", help="output xyz [<stem>_mace.xyz]")
    calc.add_argument("-device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="[auto]")
    calc.add_argument("-dtype", default="float64", choices=["float64", "float32"], help="[float64]")
    calc.add_argument("-head", default=None, help="head of a multi-head model")
    calc.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figure")
    calc.set_defaults(func=cmd_calc)

    plot = sub.add_parser("plot", parents=[common], help="re-analyse an evaluated file")
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
