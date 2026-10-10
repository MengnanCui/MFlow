# -*- coding: utf-8 -*-
"""
Is the committee's uncertainty worth anything? Calibration against real errors.

All metrics compare a predicted spread σ with the observed error δ on data that has a
reference (DFT) label. Committee spreads are usually poorly calibrated in absolute terms
(Annevelink & Viswanathan, arXiv:2308.15653), so the most useful number for active
learning is the *ranking* quality (Spearman); the calibration numbers say whether σ can
also be read as an error bar.

    spearman            rank correlation of σ and |δ| per atom            1 = perfect ranking
    miscalibration area mean |observed − expected| coverage of N(0, σ²)    0 = calibrated
                        intervals over 99 levels (Tran et al. 2020, uncertainty-toolbox)
    rms calibration err RMS of the same coverage differences
    sharpness           RMS of σ — how confident the committee claims to be
    sigma ratio         RMS(δ) / RMS(σ)  (>1: over-confident, <1: under-confident)
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np

LEVELS = np.linspace(0.01, 0.99, 99)


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), like scipy.stats.rankdata."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _z_for_central(p: np.ndarray) -> np.ndarray:
    """z with P(|N(0,1)| <= z) = p, by bisection on erf (no scipy needed)."""
    lo, hi = np.zeros_like(p), np.full_like(p, 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cover = np.array([erf(m / sqrt(2)) for m in mid])
        lo, hi = np.where(cover < p, mid, lo), np.where(cover < p, hi, mid)
    return 0.5 * (lo + hi)


def coverage_curve(delta: np.ndarray, sigma: np.ndarray, levels: np.ndarray = LEVELS):
    """Observed fraction of |δ| inside the central interval of N(0, σ²) at each level."""
    delta, sigma = np.abs(np.asarray(delta, float)), np.maximum(np.asarray(sigma, float), 1e-12)
    z = _z_for_central(levels)
    observed = np.array([np.mean(delta <= zi * sigma) for zi in z])
    return levels, observed


def calibration(delta_comp: np.ndarray, sigma_comp: np.ndarray, err_atom: np.ndarray,
                sigma_atom: np.ndarray) -> dict:
    """
    delta_comp / sigma_comp : per force component (signed error, committee std of that component)
    err_atom / sigma_atom   : per atom |ΔF| and the committee force deviation (ranking)
    """
    expected, observed = coverage_curve(delta_comp, sigma_comp)
    diff = observed - expected
    rms_sigma = float(np.sqrt(np.mean(np.square(sigma_comp))))
    rms_delta = float(np.sqrt(np.mean(np.square(delta_comp))))
    return {
        "spearman_atom": spearman(sigma_atom, err_atom),
        "miscalibration_area": float(np.mean(np.abs(diff))),
        "rms_calibration_error": float(np.sqrt(np.mean(diff ** 2))),
        "sharpness_eV_A": rms_sigma,
        "sigma_ratio": rms_delta / rms_sigma if rms_sigma > 0 else float("nan"),
        "n_components": int(np.size(delta_comp)),
        "curve": {"expected": expected.round(3).tolist(), "observed": observed.round(4).tolist()},
    }


def verdict(cal: dict) -> tuple[str, str]:
    """pass/warn/fail for 'can this committee rank structures for active learning?'."""
    from .thresholds import section
    lim = section("uq")
    rho = cal["spearman_atom"]
    if not np.isfinite(rho):
        return "warn", "not enough data to judge"
    if rho >= lim["spearman_pass"]:
        return "pass", f"σ ranks errors well (Spearman {rho:.2f})"
    if rho >= lim["spearman_warn"]:
        return "warn", f"σ ranks errors weakly (Spearman {rho:.2f})"
    return "fail", f"σ does not rank errors (Spearman {rho:.2f})"
