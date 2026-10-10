# -*- coding: utf-8 -*-
"""
Statistics for correlated time series.

block_average   Flyvbjerg & Petersen, J. Chem. Phys. 91, 461 (1989): pairwise blocking until the
                standard error stops growing; the largest SE with at least 8 blocks is reported
                (conservative). n_eff = var / SE² is the number of independent samples.
halves          equilibration check: the two halves of the production window should agree within
                their combined block errors.
"""

from __future__ import annotations

import numpy as np


def block_average(x: np.ndarray, min_blocks: int = 8) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return {"mean": float(x.mean()) if n else float("nan"), "se": float("nan"), "n_eff": n}
    best, y = 0.0, x.copy()
    while len(y) >= min_blocks:
        se = float(np.std(y, ddof=1) / np.sqrt(len(y)))
        best = max(best, se)
        y = 0.5 * (y[: len(y) // 2 * 2: 2] + y[1: len(y) // 2 * 2: 2])
    if best == 0.0:
        best = float(np.std(x, ddof=1) / np.sqrt(n))
    var = float(np.var(x, ddof=1))
    return {"mean": float(x.mean()), "se": best, "n_eff": float(var / best ** 2) if best > 0 else float(n)}


def halves(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    h = len(x) // 2
    if h < 8:
        return {}
    a, b = block_average(x[:h]), block_average(x[h:])
    err = np.hypot(a["se"], b["se"])
    z = abs(a["mean"] - b["mean"]) / err if err > 0 else 0.0
    return {"first": a["mean"], "second": b["mean"], "z": float(z)}
