"""Reward-classifier decision threshold selection (DECISIONS.md D-09).

Primary analysis: fixed, pre-registered threshold (LeRobot default 0.5).
Secondary: F-beta (beta < 1 favours precision, i.e. fewer fake successes) on the VALIDATION split using the
condition's OWN labels. This function never sees audit data; the runner only passes validation arrays.
Like LeRobot, a prediction is positive iff score > threshold (strict).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

DEFAULT_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05))


def select_threshold(
    cfg: Mapping, val_scores: Sequence[float], val_targets: Sequence[int | None]
) -> tuple[float, dict]:
    policy = cfg.get("policy", "fixed")
    if policy == "fixed":
        value = float(cfg.get("value", 0.5))
        if not 0.0 < value < 1.0:
            raise ValueError("fixed threshold must be in (0, 1)")
        return value, {"policy": "fixed", "fallback": False}
    if policy != "val_fbeta":
        raise ValueError(f"unknown threshold policy {policy!r}")
    beta = float(cfg.get("beta", 0.5))
    fallback = float(cfg.get("fallback", 0.5))
    pairs = [(float(s), int(t)) for s, t in zip(val_scores, val_targets) if t is not None and s is not None]
    n_pos = sum(t for _, t in pairs)
    if not pairs or n_pos == 0 or n_pos == len(pairs):
        return fallback, {"policy": "val_fbeta", "beta": beta, "fallback": True, "n_val": len(pairs), "n_pos": n_pos}
    s = np.array([p[0] for p in pairs])
    t = np.array([p[1] for p in pairs])
    best_thr, best_f = fallback, -1.0
    for thr in DEFAULT_GRID:
        pred = s > thr
        tp = int(np.sum(pred & (t == 1)))
        fp = int(np.sum(pred & (t == 0)))
        fn = int(np.sum(~pred & (t == 1)))
        if tp == 0:
            f = 0.0
        else:
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        if f >= best_f:  # ties -> larger (more conservative) threshold, since the grid is ascending
            best_thr, best_f = float(thr), f
    return best_thr, {"policy": "val_fbeta", "beta": beta, "fallback": False, "n_val": len(pairs), "n_pos": n_pos,
                      "val_fbeta": best_f}
