# -*- coding: utf-8 -*-
"""
Selection — which explored structures are worth a DFT calculation.

1. Committee re-scoring (round >= 1): Fdev_max per frame, DP-GEN's max force deviation
   (|std over models of F_i|, max over atoms). Classified like DP-GEN:
       accurate   Fdev_max <  lo        the model already knows it
       candidate  lo <= Fdev_max < hi   label these
       failed     Fdev_max >= hi        usually unphysical, skipped (counted)
   trust modes:  fixed     lo / hi in eV/Å (campaign.yaml or thresholds.yaml, DP-GEN-style 0.05 / 0.15)
                 auto      lo = Q3 + 1.5·IQR of Fdev_max on the current training set (MFlow's
                           heuristic, Tukey fence — not a DP-GEN feature); hi = max(3·lo, fixed hi)
                 relative  Fdev / (|F| + floor) per atom, max over atoms, with relative lo / hi
                 adaptive-lower (v0.3, dpgen2) lo follows the explored deviations so that
                           max(numb, rate × N) frames are candidates; once lo is stable for
                           `consecutive` rounds it is frozen and hi = lo + offset
   Without a committee (v0.3, select.uncertainty): descriptor = distance to the nearest training
   frame above the training set's own leave-one-out quantile; quests = any atom with dH > 0;
   none = every frame is a candidate, FPS only. See mflow/al/coverage.py.
   Recalibration (fixed windows): deviations are multiplied by RMS(error)/RMS(σ) from the
   committee's test-set calibration when that ratio is > 1 (variance scaling).
   Round 0 has no committee: every explored frame is a candidate and only diversity counts.
2. Diversity: farthest-point sampling on per-frame MACE descriptors (mean over atoms,
   standardised), conditional on the training set — each pick is the frame farthest from
   everything already labelled or picked. Caps the batch at max_per_round.
"""

from __future__ import annotations

import numpy as np
import ase.io

from ..log import LOG, table
from ..model import pick_device


def fps(X: np.ndarray, n: int, X_ref: np.ndarray | None = None) -> list[int]:
    """Greedy farthest-point sampling; with X_ref the first pick is already far from the reference set."""
    X = np.asarray(X, float)
    if len(X) == 0 or n <= 0:
        return []
    d = np.full(len(X), np.inf)
    if X_ref is not None and len(X_ref):
        for r in X_ref:
            d = np.minimum(d, ((X - r) ** 2).sum(1))
        picks = []
    else:
        picks = [0]
        d = ((X - X[0]) ** 2).sum(1)
    while len(picks) < min(n, len(X)):
        i = int(np.argmax(d))
        if d[i] <= 0 and picks:
            break                                   # only duplicates left
        picks.append(i)
        d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    return picks


def descriptors(frames: list, model: str) -> np.ndarray:
    """Per-frame mean of MACE invariant descriptors (single model)."""
    from ..model import build_calculator
    calc = build_calculator(model, pick_device("auto"), "float64", None)
    return np.array([np.asarray(calc.get_descriptors(a, invariants_only=True)).mean(0) for a in frames])


def committee_scores(frames: list, models: list, relative_floor: float | None = None) -> np.ndarray:
    from ..calc import evaluate_committee
    work = [a.copy() for a in frames]
    evaluate_committee(work, models, pick_device("auto"), "float64", "al_", None)
    if relative_floor is None:
        return np.array([a.info["al_Fdev_max"] for a in work])
    return np.array([float(np.max(a.arrays["al_Fdev"] / (np.linalg.norm(a.arrays["al_forces"], axis=1)
                                                          + relative_floor))) for a in work])


def trust_window(camp, models: list, train_frames: list, settings: dict) -> tuple[float, float, str]:
    from ..thresholds import section
    al = section("al")
    mode = settings.get("trust", "fixed")
    if mode == "relative":
        return al["relative_lo"], al["relative_hi"], mode
    lo, hi = float(settings["lo"]), float(settings["hi"])
    if mode == "auto" and train_frames:
        sample = train_frames[:: max(1, len(train_frames) // 300)]
        dev = committee_scores(sample, models)
        q1, q3 = np.quantile(dev, [0.25, 0.75])
        auto_lo = float(q3 + 1.5 * (q3 - q1))
        if auto_lo >= hi:
            LOG(f"[warn]trust auto: the committee already disagrees by {auto_lo:.3f} eV/Å on its own training "
                f"data (above the fixed hi {hi:.3f}) — the models are not converged, so an auto window would call "
                f"everything 'accurate'. Using the fixed window {lo:.3g}–{hi:.3g} eV/Å; train longer.[/warn]")
            return lo, hi, "fixed (auto rejected: committee not converged)"
        lo, hi = auto_lo, max(3 * auto_lo, hi)
        LOG(f"[dim]trust auto: Fdev_max on training set Q1 {q1:.3f} Q3 {q3:.3f} eV/Å → lo {lo:.3f}, hi {hi:.3f}[/dim]")
    return lo, hi, mode


def recalibration(camp, round_: int) -> float:
    """
    Variance scaling: multiply the committee deviation by RMS(error)/RMS(σ) measured on the
    test set (committee report of this round), so a fixed eV/Å window means what it says.
    Fine-tuned members start from the same weights and tend to be over-confident (ratio > 1).
    """
    import json
    for path in sorted(camp.round_dir(round_, "train").glob("*_committee_report.json")):
        cal = (json.loads(path.read_text()).get("test") or {}).get("calibration") or {}
        ratio = cal.get("sigma_ratio")
        if ratio and np.isfinite(ratio) and ratio > 1.0:
            LOG(f"[warn]committee is over-confident on the test set (RMS error / RMS σ = {ratio:.1f}): "
                f"deviations are scaled by {ratio:.1f} before the trust window[/warn]")
            return float(ratio)
    return 1.0


def uncertainty_mode(camp, models: list) -> str:
    want = camp.settings["select"].get("uncertainty", "auto")
    if not models:
        return "seed"                                   # round 0: diversity only
    if want == "auto":
        return "committee" if len(models) >= 2 else "descriptor"
    if want == "committee" and len(models) < 2:
        LOG("[warn]select.uncertainty committee needs ≥ 2 models — using descriptor distance[/warn]")
        return "descriptor"
    if want == "quests":
        from .coverage import QUESTS_HINT, have_quests
        if not have_quests():
            LOG(f"[warn]select.uncertainty quests: QUESTS is not installed ({QUESTS_HINT}) — using descriptor "
                f"distance[/warn]")
            return "descriptor"
    return want


def adaptive_window(camp, score: np.ndarray) -> tuple[float, float, str]:
    from ..thresholds import section
    from .convergence import adaptive_lo, lo_is_stable, selects
    th = section("al")
    frozen = camp.state.get("trust_frozen_lo")
    if frozen is not None:
        return float(frozen), float(frozen) + th["adaptive_hi_offset_eV_A"], "adaptive-lower (frozen)"
    hi = float(camp.settings["select"].get("hi_adaptive") or th["adaptive_hi_eV_A"])
    lo = adaptive_lo(score, hi, th["adaptive_numb_candi"], th["adaptive_rate_candi"])
    history = [x["trust_lo"] for x in selects(camp.ledger()) if x.get("trust_mode") == "adaptive-lower"] + [lo]
    if lo_is_stable(history, th["adaptive_conv_tol_eV_A"], int(th["consecutive"])):
        camp.save_state(trust_frozen_lo=lo)
        LOG(f"[ok]adaptive-lower: lo stable within {th['adaptive_conv_tol_eV_A']} eV/Å over "
            f"{th['consecutive']} rounds → frozen at {lo:.3f} eV/Å[/ok] (hi = lo + {th['adaptive_hi_offset_eV_A']})")
    else:
        LOG(f"[dim]adaptive-lower: lo {lo:.3f} eV/Å (history {', '.join(f'{x:.3f}' for x in history)})[/dim]")
    return lo, hi, "adaptive-lower"


def by_region(frames: list, accurate, candidate, failed) -> dict:
    regs = {}
    for i, a in enumerate(frames):
        r = regs.setdefault(a.info.get("mflow_region", "all"), {"n": 0, "accurate": 0, "candidate": 0, "failed": 0})
        r["n"] += 1
        r["accurate"] += int(accurate[i])
        r["candidate"] += int(candidate[i])
        r["failed"] += int(failed[i])
    return regs


def run_selection(camp, round_: int, frames: list) -> tuple[list, dict]:
    import json
    from ..thresholds import section
    from .convergence import current_region
    s = camp.settings
    sel = s["select"]
    th = section("al")
    models = camp.committee()
    out_dir = camp.round_dir(round_, "select")
    region = current_region(camp)["name"]
    stats = {"n_explored": len(frames), "region": region}
    state = camp.state
    train = ase.io.read(state["dataset"], ":") if state.get("dataset") and models else []
    mode = uncertainty_mode(camp, models)
    stats["uncertainty"] = mode
    model = camp.explorer_model()
    cache = {}

    def X_all():                                        # descriptors of every explored frame, computed once
        if "X" not in cache:
            cache["X"] = descriptors(frames, model)
            ref = train[:: max(1, len(train) // 500)]
            cache["X_train"] = descriptors(ref, model) if ref else None
        return cache["X"], cache["X_train"]

    coverage = {}
    if mode == "committee":
        lo, hi, tmode = trust_window(camp, models, train, sel) if sel.get("trust") != "adaptive-lower" else (0, 0, "")
        floor = section("al")["relative_floor_eV_A"] if tmode == "relative" else None
        work = [a.copy() for a in frames]
        from ..calc import evaluate_committee
        evaluate_committee(work, models, pick_device("auto"), "float64", "al_", None)
        if floor is None:
            score = np.array([a.info["al_Fdev_max"] for a in work])
        else:
            score = np.array([float(np.max(a.arrays["al_Fdev"] / (np.linalg.norm(a.arrays["al_forces"], axis=1)
                                                                    + floor))) for a in work])
        wo0 = [a.info.get("al_Fdev_max_wo0") for a in work]
        if sel.get("trust") == "adaptive-lower":
            lo, hi, tmode = adaptive_window(camp, score)
        scale = recalibration(camp, round_) if sel.get("recalibrate", True) and tmode.startswith("fixed") else 1.0
        score = score * scale
        stats["recalibration"] = scale
        if all(v is not None for v in wo0) and wo0:
            stats["dev_median_wo_driver"] = float(np.median(np.asarray(wo0) * scale))
        accurate, candidate, failed = score < lo, (score >= lo) & (score < hi), score >= hi
        label = f"committee ({tmode}, lo {lo:.3g}, hi {hi:.3g})"
    elif mode in ("descriptor", "quests"):
        if mode == "descriptor":
            from .coverage import descriptor_scores
            X, X_train = X_all()
            score, thr = descriptor_scores(X, X_train, th["ood_quantile"]) if X_train is not None else \
                (np.zeros(len(frames)), 0.0)
            tmode, lo = f"descriptor distance (q{th['ood_quantile']:.2f})", thr
            coverage.update({"ood_fraction": float((score > thr).mean()) if len(score) else 0.0,
                             "ood_threshold": thr})
        else:
            from .coverage import quests_scores
            q = quests_scores(frames, train)
            score, lo, tmode = q["dH_max"], 0.0, "QUESTS dH > 0"
            coverage.update({k: v for k, v in q.items() if k != "dH_max"})
        hi = float("inf")
        accurate, candidate, failed = score <= lo, score > lo, np.zeros(len(score), bool)
        stats["recalibration"] = 1.0
        label = f"{tmode}, threshold {lo:.3g}"
    else:
        score = None

    if score is not None:
        for a, v in zip(frames, score):
            a.info["mflow_dev"] = float(v)
        stats.update({"trust_mode": tmode, "trust_lo": float(lo), "trust_hi": float(hi),
                      "n_accurate": int(accurate.sum()), "n_candidate": int(candidate.sum()),
                      "n_failed": int(failed.sum()), "candidate_fraction": float(candidate.mean()) if len(score) else 0.0,
                      "dev_median": float(np.median(score)) if len(score) else None,
                      "regions": by_region(frames, accurate, candidate, failed)})
        pool = [f for f, c in zip(frames, candidate) if c]
        cap = int(sel["max_per_round"])
        if not pool and failed.mean() > 0.5:
            LOG(f"[warn]no candidate and {failed.mean():.0%} of the explored frames are above hi: the model is far "
                f"from this region, not the structures unphysical. Labelling a diverse subset of the 'failed' "
                f"frames instead (DP-GEN would skip them); check that MD did not explode.[/warn]")
            pool = [f for f, x in zip(frames, failed) if x]
            stats["pool"] = "failed (fallback: model not ready)"
        LOG(table(f"round {round_} · {label}", ["", "frames", "fraction"],
                  [(name, f"{int(m.sum()):,}", f"{m.mean() * 100:.1f} %") for name, m in
                   (("accurate", accurate), ("candidate", candidate), ("failed", failed))],
                  styles=["key"]))
    elif mode == "none":
        pool = list(frames)
        cap = int(sel["max_per_round"])
        stats.update({"trust_mode": "none (FPS only)", "n_candidate": len(pool)})
    else:
        pool = list(frames)
        cap = int(s["seed"]["n_select"])
        stats.update({"trust_mode": "none (round 0: diversity only)", "n_candidate": len(pool)})

    # coverage for the L2 criterion (cheap when the descriptors are needed anyway)
    cov_mode = sel.get("coverage", "auto")
    if models and cov_mode != "off" and not coverage:
        from .coverage import QUESTS_HINT, descriptor_scores, have_quests, quests_scores
        try:
            if cov_mode == "quests" or (cov_mode == "auto" and have_quests()):
                if have_quests():
                    q = quests_scores(frames, train)
                    coverage.update({k: v for k, v in q.items() if k != "dH_max"})
                else:
                    LOG(f"[warn]select.coverage quests: QUESTS not installed — {QUESTS_HINT}[/warn]")
            if not coverage:
                X, X_train = X_all()
                if X_train is not None:
                    d, thr = descriptor_scores(X, X_train, th["ood_quantile"])
                    coverage.update({"ood_fraction": float((d > thr).mean()) if len(d) else 0.0, "ood_threshold": thr})
        except Exception as exc:
            LOG(f"[warn]coverage skipped: {type(exc).__name__}: {exc}[/warn]")
    if coverage:
        stats.update({k: coverage[k] for k in ("ood_fraction", "dH_pos_fraction") if k in coverage})
        (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        LOG("coverage: " + " · ".join(f"{k} {v:.3g}" for k, v in coverage.items() if isinstance(v, float)))

    picks = list(range(len(pool)))
    if len(pool) > cap and sel.get("dedupe", "fps") == "fps":
        if "X" in cache:
            idx = {id(f): i for i, f in enumerate(frames)}
            X, X_ref = cache["X"][[idx[id(f)] for f in pool]], cache["X_train"]
        else:
            X = descriptors(pool, model)
            ref = train[:: max(1, len(train) // 500)]
            X_ref = descriptors(ref, model) if ref else None
        mu, sd = X.mean(0), X.std(0) + 1e-12
        picks = fps((X - mu) / sd, cap, None if X_ref is None else (X_ref - mu) / sd)
    elif len(pool) > cap:
        picks = sorted(np.random.default_rng(round_).choice(len(pool), cap, replace=False).tolist())
    selected = [pool[i] for i in picks]
    for k, a in enumerate(selected):
        a.info.update({"mflow_id": f"r{round_:02d}_c{k:05d}", "mflow_round": round_,
                       "config_type": a.info.get("config_type", f"al_{a.info.get('mflow_source', 'x')}")})
    stats["n_selected"] = len(selected)
    ase.io.write(out_dir / "selected.xyz", selected, format="extxyz")
    LOG(f"round {round_}: [hi]{len(selected)}[/hi] selected from {len(pool)} candidate(s) "
        f"({len(frames)} explored) → {out_dir / 'selected.xyz'}")
    return selected, stats
