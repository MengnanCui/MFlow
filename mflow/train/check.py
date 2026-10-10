# -*- coding: utf-8 -*-
"""
mflow check — is this dataset fit to train on, and what is missing?

Every rule is a pure function of the frames (plus, optionally, a foundation model's
zero-shot prediction) so the same checks run standalone and as the gate in front of
`mflow train`.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import ase.io
from ase.data import chemical_symbols

from .. import __version__
from ..data import composition, find_ref, fix_prefix, group_of, has_labels, keys, read_dataset
from ..log import LOG, elapsed
from ..report import Report
from ..style import COLORS, plt

# thresholds — one place, so the json and the docs agree
F_WARN, F_FAIL = 50.0, 200.0          # eV/Å, a single force component magnitude
D_MIN = 0.6                           # Å, closest allowed pair distance
E_OUTLIER_Z = 6.0                     # robust z-score (MAD) of the E0-fitted energy residual
E0_OFFSET_WARN = 0.4                  # eV/atom raw zero-shot offset that hints at a DFT mismatch
MIN_FRAMES, MIN_GROUP = 50, 10


def stress_key(prefix: str) -> str:
    return f"{prefix}stress"


def frame_hash(atoms) -> str:
    """Geometry fingerprint — identical structures (to 1e-4 Å) share it."""
    h = hashlib.sha1()
    h.update(np.asarray(atoms.numbers, np.int64).tobytes())
    h.update(np.round(atoms.get_positions(), 4).tobytes())
    h.update(np.round(np.array(atoms.cell), 4).tobytes())
    return h.hexdigest()


def min_distance(atoms, cutoff: float = 2.5) -> float:
    """Closest pair (periodic images included); cutoff when nothing is closer."""
    from ase.neighborlist import neighbor_list
    if len(atoms) < 2 and not np.any(atoms.pbc):
        return float("inf")
    d = neighbor_list("d", atoms, cutoff)
    return float(d.min()) if d.size else cutoff


def e0_residuals(frames: list, prefix: str) -> np.ndarray:
    """
    Per-atom residual after a per-element linear fit E ≈ Σ n_j μ_j (meV/atom). Refitted once
    without the gross outliers, so one broken structure cannot drag the fit onto the good ones.
    """
    C, _ = composition(frames)
    E = np.array([float(a.info[keys(prefix)[0]]) for a in frames])
    keep = np.ones(len(E), bool)
    for _ in range(2):
        mu, *_ = np.linalg.lstsq(C[keep], E[keep], rcond=None)
        resid = (E - C @ mu) / C.sum(1) * 1000
        keep = robust_z(resid) <= E_OUTLIER_Z
        if keep.sum() < C.shape[1] + 2:
            break
    return resid


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return np.abs(x - med) / mad if mad > 0 else np.zeros_like(x)


def is_isolated_atom(atoms) -> bool:
    return len(atoms) == 1 and (atoms.info.get("config_type") == "IsolatedAtom" or not np.any(atoms.pbc))


# ============================================================================ #
#  the rules
# ============================================================================ #

def check_frames(frames: list, ref: str, rep: Report, group_key: str | None = None,
                 has_test: bool = False) -> dict:
    """All model-free checks. Returns per-frame flags for the figure and the json."""
    n = len(frames)
    ekey, fkey = keys(ref)
    skey = stress_key(ref)
    flags = {"nan": [], "f_warn": [], "f_fail": [], "close": [], "duplicate": [], "e_outlier": []}

    # -- labels ------------------------------------------------------------- #
    has_e = np.array([ekey in a.info for a in frames])
    has_f = np.array([fkey in a.arrays for a in frames])
    has_s = np.array([skey in a.info for a in frames])
    if not has_e.all() or not has_f.all():
        rep.check("labels", "fail", f"{ekey}: {has_e.sum()}/{n} · {fkey}: {has_f.sum()}/{n}",
                  suggest=f"every structure needs {ekey} and {fkey}; drop or relabel the others")
    else:
        rep.check("labels", "pass", f"{ekey} / {fkey} on all {n} structures")
    if has_s.any() and not has_s.all():
        rep.check("stress", "warn", f"{skey} on {has_s.sum()}/{n} structures",
                  suggest="stress on only some structures: complete it or train with stress_weight 0")
    else:
        rep.check("stress", "pass", f"{skey} on {'all' if has_s.all() else 'no'} structures")

    labelled = [i for i in range(n) if has_e[i] and has_f[i]]

    # -- numbers ------------------------------------------------------------ #
    fmax = np.full(n, np.nan)
    for i in labelled:
        a = frames[i]
        e, f = float(a.info[ekey]), np.asarray(a.arrays[fkey], float)
        if not np.isfinite(e) or not np.all(np.isfinite(f)):
            flags["nan"].append(i)
            continue
        fmax[i] = np.abs(f).max()
        if fmax[i] > F_FAIL:
            flags["f_fail"].append(i)
        elif fmax[i] > F_WARN:
            flags["f_warn"].append(i)
    rep.check("finite labels", "fail" if flags["nan"] else "pass",
              f"{len(flags['nan'])} structures with NaN/inf" if flags["nan"] else "no NaN / inf",
              value=flags["nan"][:50], suggest="remove the NaN/inf structures (a failed DFT run?)")
    status = "fail" if flags["f_fail"] else ("warn" if flags["f_warn"] else "pass")
    rep.check("force magnitude", status,
              f"max |F_i| {np.nanmax(fmax):.1f} eV/Å · >{F_WARN:.0f}: {len(flags['f_warn'])} · "
              f">{F_FAIL:.0f}: {len(flags['f_fail'])}" if labelled else "no labels",
              value=(flags["f_fail"] + flags["f_warn"])[:50],
              suggest=f"forces above {F_WARN:.0f} eV/Å are usually unconverged SCF or overlapping atoms: "
                      "inspect them, then drop or recompute")

    # -- geometry ------------------------------------------------------------ #
    dmin = np.array([min_distance(a) for a in frames])
    flags["close"] = [int(i) for i in np.where(dmin < D_MIN)[0]]
    rep.check("closest pair", "fail" if flags["close"] else "pass",
              f"min distance {np.min(dmin):.2f} Å" + (f" · {len(flags['close'])} below {D_MIN} Å"
                                                       if flags["close"] else ""),
              value=flags["close"][:50], suggest=f"structures with atoms closer than {D_MIN} Å are broken")
    bad_cell = [i for i, a in enumerate(frames) if np.any(a.pbc) and abs(a.get_volume()) < 1e-6]
    rep.check("cell", "fail" if bad_cell else "pass",
              f"{len(bad_cell)} periodic structures with zero volume" if bad_cell else "cells valid",
              value=bad_cell[:50], suggest="periodic structures need a 3D cell")

    seen, dup = {}, []
    for i, a in enumerate(frames):
        h = frame_hash(a)
        if h in seen:
            dup.append(i)
        seen.setdefault(h, i)
    flags["duplicate"] = dup
    rep.check("duplicates", "warn" if dup else "pass",
              f"{len(dup)} duplicate geometries" if dup else "no duplicates", value=dup[:50],
              suggest="duplicates leak between train/valid/test and inflate their weight: deduplicate")

    # -- energies ------------------------------------------------------------ #
    clean = [i for i in labelled if i not in flags["nan"]]
    if len(clean) >= 5:
        sub = [frames[i] for i in clean]
        resid = e0_residuals(sub, ref)
        z = robust_z(resid)
        flags["e_outlier"] = [clean[j] for j in np.where(z > E_OUTLIER_Z)[0]]
        rep.check("energy outliers", "warn" if flags["e_outlier"] else "pass",
                  f"{len(flags['e_outlier'])} structures > {E_OUTLIER_Z:.0f} MAD from the per-element fit"
                  if flags["e_outlier"] else "energies consistent with a per-element fit",
                  value=flags["e_outlier"][:50],
                  suggest="energy outliers often mean mixed DFT settings (functional, +U, ENCUT): check them")
        rep.data["e0_fit_residual_meV_per_atom"] = {"std": float(np.std(resid)), "max": float(np.abs(resid).max())}

    # -- coverage ------------------------------------------------------------ #
    isolated = [i for i, a in enumerate(frames) if is_isolated_atom(a)]
    _, zs = composition(frames)
    els_iso = sorted({chemical_symbols[int(frames[i].numbers[0])] for i in isolated})
    els = [chemical_symbols[z] for z in zs]
    missing_e0 = [e for e in els if e not in els_iso]
    rep.check("isolated atoms (E0)", "pass" if not missing_e0 else "warn",
              "E0 for every element" if not missing_e0 else f"no IsolatedAtom for {' '.join(missing_e0)}",
              suggest="no isolated-atom energies: training will use --E0s estimated (fine for fine-tuning; "
                      "for scratch training compute them with the same DFT settings)")

    rep.check("size", "warn" if n < MIN_FRAMES else "pass", f"{n} structures",
              suggest=f"fewer than {MIN_FRAMES} structures: fine-tune a foundation model rather than train "
                      "from scratch, and expect to add data")
    groups = group_of(frames, group_key) if group_key else None
    if groups is not None:
        names, counts = np.unique(groups, return_counts=True)
        small = [str(g) for g, c in zip(names, counts) if c < MIN_GROUP]
        rep.data["groups"] = {str(g): int(c) for g, c in zip(names, counts)}
        rep.check("group sizes", "warn" if small else "pass",
                  f"{len(names)} groups, smallest {counts.min()}" + (f" · thin: {', '.join(small)}" if small else ""),
                  value=small, suggest=f"groups under {MIN_GROUP} structures are barely constrained: "
                                       "add configurations of that kind")
    if not has_test:
        rep.check("test set", "warn", "no separate test set",
                  suggest="keep a test set apart from training: pass -test, or -split 0.1 to carve one out")

    rep.data.update({"n_structures": n, "elements": els, "n_isolated_atoms": len(isolated),
                     "atoms_per_element": dict(zip(els, composition(frames)[0].sum(0).astype(int).tolist())),
                     "has_stress": bool(has_s.all())})
    return {"fmax": fmax, "dmin": dmin, **flags}


def zero_shot(frames: list, ref: str, model: str, head: str | None, device: str, dtype: str,
              rep: Report, batch: int = 16) -> dict | None:
    """Foundation model straight on the data: coverage, error level and the E0 offset."""
    from ..calc import compare, run_mace
    from ..model import load_model

    try:
        net, _ = load_model(model, device, dtype)
        model_z = {int(z) for z in net.atomic_numbers}
        heads = list(getattr(net, "heads", []) or [])
        del net
    except Exception as exc:
        rep.check("foundation model", "warn", f"could not load {model}: {type(exc).__name__}: {exc}")
        return None
    data_z = {int(z) for a in frames for z in a.numbers}
    missing = sorted(data_z - model_z)
    if missing:
        rep.check("element coverage", "fail", f"model lacks {' '.join(chemical_symbols[z] for z in missing)}",
                  suggest="the foundation model has never seen these elements: pick another model or train "
                          "from scratch")
        return None
    rep.check("element coverage", "pass", f"model covers all {len(data_z)} elements")
    if len(heads) > 1 and head is None:
        head = heads[0]
        LOG(f"[dim]multi-head model, zero-shot uses head {head} (heads: {', '.join(heads)})[/dim]")

    work = [a.copy() for a in frames]
    for a in work:
        a.calc = None
    work = run_mace(work, spec=model, batch_size=batch, device=device, dtype=dtype, prefix="zs_", head=head)
    errors, series = compare(work, "zs_", ref, "fit")
    e_zs = np.array([float(a.info["zs_energy"]) for a in work])
    e_ref = np.array([float(a.info[keys(ref)[0]]) for a in work])
    offset = float(np.mean((e_zs - e_ref) / series["n_atoms"]))
    rep.data["zero_shot"] = {"model": model, "head": head, "errors": errors, "raw_offset_eV_per_atom": offset}
    f_rmse = errors.get("forces", {}).get("RMSE", float("nan"))
    rep.check("zero-shot error", "pass",
              f"E {errors['energy']['RMSE']:.1f} meV/atom (E0 fitted) · F {f_rmse:.0f} meV/Å")
    rep.check("zero-shot E0 offset", "warn" if abs(offset) > E0_OFFSET_WARN else "pass",
              f"raw offset {offset:+.3f} eV/atom",
              suggest="large raw energy offset: your DFT settings likely differ from the foundation model's "
                      "(MP PBE+U / OMat24 / MatPES); use --E0s estimated, or multihead fine-tuning")
    return {"errors": errors, "series": series, "frames": work}


# ============================================================================ #
#  split / figure / command
# ============================================================================ #

def split_frames(frames: list, fraction: float, groups: np.ndarray | None, seed: int = 0):
    """Stratified by group when there is one; isolated atoms always stay in training."""
    rng = np.random.default_rng(seed)
    labels = groups if groups is not None else np.array(["all"] * len(frames))
    test = []
    for g in dict.fromkeys(labels):
        idx = [i for i in np.where(labels == g)[0] if not is_isolated_atom(frames[i])]
        k = int(round(len(idx) * fraction))
        if len(idx) >= 2 and k == 0:
            k = 1
        test += rng.choice(idx, size=k, replace=False).tolist() if k else []
    test = set(test)
    return [a for i, a in enumerate(frames) if i not in test], [a for i, a in enumerate(frames) if i in test]


def make_check_plot(frames, ref, flags, zs, outfile):
    ekey = keys(ref)[0]
    n = np.array([len(a) for a in frames], float)
    e = np.array([float(a.info.get(ekey, np.nan)) for a in frames]) / n
    ncol = 3 if zs else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.3))
    axes[0].hist(e[np.isfinite(e)], bins=40, color=COLORS[0], alpha=0.75, edgecolor="black", linewidth=1.0)
    axes[0].set(xlabel="Energy (eV/atom)", ylabel="Counts")
    f = flags["fmax"][np.isfinite(flags["fmax"])]
    axes[1].hist(np.log10(np.clip(f, 1e-3, None)), bins=40, color=COLORS[1], alpha=0.75,
                 edgecolor="black", linewidth=1.0)
    axes[1].axvline(np.log10(F_WARN), color="black", ls="--", lw=2)
    axes[1].set(xlabel=r"log$_{10}$ max |F$_i$| (eV/$\mathrm{\AA}$)", ylabel="Counts")
    if zs:
        s, err = zs["series"], zs["errors"]["energy"]
        axes[2].scatter(s["e_ref"], s["e_pred"], s=30, color=COLORS[0], alpha=0.7, edgecolors=COLORS[0])
        lo, hi = min(s["e_ref"].min(), s["e_pred"].min()), max(s["e_ref"].max(), s["e_pred"].max())
        axes[2].plot([lo, hi], [lo, hi], "k--", lw=2)
        axes[2].set(xlabel="DFT energy (eV/atom)", ylabel="zero-shot (eV/atom, E0 fit)",
                    title=f"RMSE {err['RMSE']:.1f} meV/atom")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


def run_check(args, frames: list | None = None) -> Report:
    """Shared by `mflow check` and the pre-flight of `mflow train`."""
    t0 = time.perf_counter()
    frames = frames if frames is not None else read_dataset(args.input, args.index)
    ref = fix_prefix(args.ref_prefix) if args.ref_prefix is not None else find_ref(frames, exclude="zs_")
    stem = Path(args.input).stem
    rep = Report("check", input=str(Path(args.input).resolve()))
    if ref is None:
        rep.check("labels", "fail", "no energy/forces labels found (tried REF_, dft_, energy, ...)",
                  suggest="name the labels REF_energy / REF_forces (or pass -ref PREFIX)")
        return rep
    rep.data["ref_prefix"] = ref
    LOG(f"reference labels [key]{keys(ref)[0]}[/key] / [key]{keys(ref)[1]}[/key]")

    group_key = args.group or ("config_type" if all("config_type" in a.info for a in frames) else None)
    flags = check_frames(frames, ref, rep, group_key, has_test=bool(args.test or args.split))
    zs = None
    if args.model and args.model != "none" and not rep.failed:
        from ..model import pick_device
        zs = zero_shot(frames, ref, args.model, args.head, pick_device(args.device), args.dtype, rep)
    rep.data["flags"] = {k: v for k, v in flags.items() if k not in ("fmax", "dmin")}

    if args.split and not rep.failed:
        train, test = split_frames(frames, args.split, group_of(frames, group_key), args.seed)
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", train), ("test", test)):
            path = out / f"{stem}_{name}.xyz"
            ase.io.write(path, part, format="extxyz")
            rep.data[f"{name}_file"] = str(path.resolve())
        LOG(f"split: [key]{rep.data['train_file']}[/key] ({len(train)}) · "
            f"[key]{rep.data['test_file']}[/key] ({len(test)})")

    if not getattr(args, "no_plot", False):
        make_check_plot(frames, ref, flags, zs, str(Path(args.outdir) / f"{stem}_check.png"))
    rep.data["walltime_s"] = round(time.perf_counter() - t0, 1)
    return rep


def cmd_check(args) -> int:
    LOG.rule(f"MFlow {__version__} · check")
    LOG.kv({"input": args.input, "test": args.test or "—", "reference": args.ref_prefix or "auto",
            "zero-shot model": args.model or "—", "head": args.head or "—", "group by": args.group or "auto",
            "split": args.split or "—", "outdir": args.outdir})
    LOG.rule("dataset")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rep = run_check(args)
    if args.test:
        test = read_dataset(args.test)
        ref = rep.data.get("ref_prefix", "")
        if not all(has_labels(a, ref) for a in test):
            rep.check("test labels", "fail", f"test set lacks {keys(ref)[0]} / {keys(ref)[1]}")
        train_h = {frame_hash(a) for a in read_dataset(args.input, args.index)}
        leak = sum(frame_hash(a) in train_h for a in test)
        rep.check("train/test overlap", "fail" if leak else "pass",
                  f"{leak} test structures also in training" if leak else "no overlap",
                  suggest="remove the overlapping structures from the test set")
    LOG.rule("verdict")
    rep.show()
    rep.write(Path(args.outdir) / f"{Path(args.input).stem}_check.json")
    LOG(f"[dim]total {elapsed(rep.data.get('walltime_s', 0))}[/dim]")
    return 0 if not rep.failed else 2

