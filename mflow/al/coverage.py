# -*- coding: utf-8 -*-
"""
Coverage and out-of-distribution scores — model-agnostic uncertainty for selection without a
committee, and the L2 coverage criterion.

    descriptor   distance of each frame (mean MACE descriptor, standardised on the training set)
                 to its nearest training frame; OOD when above the ood_quantile of the training
                 set's own leave-one-out nearest-neighbour distances (thresholds.yaml al)
    quests       QUESTS differential entropy dH (Schwalbe-Koda et al., Nat. Commun. 16, 4014 (2025);
                 github.com/dskoda/quests, BSD-3) of every atomic environment against the training
                 set; a frame is new when any of its atoms has dH > 0 — a threshold that needs no
                 calibration. Optional: pip install quests
"""

from __future__ import annotations

import numpy as np

QUESTS_HINT = "pip install quests   (BSD-3; github.com/dskoda/quests)"


def nn_distance(X: np.ndarray, Y: np.ndarray, exclude_self: bool = False, chunk: int = 2048) -> np.ndarray:
    """Euclidean distance from every row of X to its nearest row of Y."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    out = np.empty(len(X))
    y2 = (Y ** 2).sum(1)
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        d2 = (x ** 2).sum(1)[:, None] + y2[None, :] - 2 * x @ Y.T
        if exclude_self:
            idx = np.arange(s, s + len(x))
            d2[np.arange(len(x)), idx] = np.inf
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def ood_threshold(X_train: np.ndarray, quantile: float) -> float:
    """quantile of the leave-one-out nearest-neighbour distances inside the training set."""
    if len(X_train) < 2:
        return 0.0
    return float(np.quantile(nn_distance(X_train, X_train, exclude_self=True), quantile))


def descriptor_scores(X: np.ndarray, X_train: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    """(nearest-training-frame distance per frame, OOD threshold), both in standardised units."""
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-12
    Xs, Ts = (X - mu) / sd, (X_train - mu) / sd
    return nn_distance(Xs, Ts), ood_threshold(Ts, quantile)


def have_quests() -> bool:
    try:
        import quests  # noqa: F401
        return True
    except ImportError:
        return False


def quests_scores(frames: list, train: list, h: float = 0.015, approx_above: int = 50000) -> dict:
    """Per-frame max dH over atoms, the fraction of new frames, and the training-set entropy."""
    from quests.descriptor import get_descriptors
    from quests.entropy import delta_entropy, entropy
    x = get_descriptors(train)
    y_all = get_descriptors(frames)
    counts = [len(a) for a in frames]
    if len(x) > approx_above:
        from quests.entropy import approx_delta_entropy
        dH = approx_delta_entropy(y_all, x, h=h)
    else:
        dH = delta_entropy(y_all, x, h=h)
    per_frame, s = [], 0
    for n in counts:
        per_frame.append(float(np.max(dH[s:s + n])))
        s += n
    per_frame = np.asarray(per_frame)
    sample = x if len(x) <= 20000 else x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    return {"dH_max": per_frame, "dH_pos_fraction": float((per_frame > 0).mean()) if len(per_frame) else 0.0,
            "entropy_train": float(entropy(sample, h=h)), "n_env_train": int(len(x)), "bandwidth": h}
