"""Binary classification metrics with explicit handling of missing/excluded labels.

Conventions:
* Positive class = SUCCESS (target 1). A false positive is a "fake success".
* Pairs where the reference or prediction is None (ambiguous/invalid/no label) are excluded and counted.
* A rate whose denominator is zero is reported as None ("unavailable"), never as 0 or NaN.
* Frame/keyframe metrics are correlated within an episode, so uncertainty intervals use an episode-cluster
  bootstrap (resampling whole episodes), not a frame-level binomial interval.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import numpy as np

Z95 = 1.959963984540054


@dataclass(frozen=True)
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def __add__(self, other: "BinaryCounts") -> "BinaryCounts":
        return BinaryCounts(self.tp + other.tp, self.fp + other.fp, self.tn + other.tn, self.fn + other.fn)

    def to_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn, "n": self.n}


@dataclass(frozen=True)
class CountResult:
    counts: BinaryCounts
    n_pairs: int
    n_ref_missing: int  # reference label excluded (ambiguous/invalid/none); counted, not hidden
    n_pred_missing: int  # prediction missing (e.g. VLM error or ambiguous)


def count_binary(y_true: Sequence[int | None], y_pred: Sequence[int | None]) -> CountResult:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    tp = fp = tn = fn = 0
    ref_missing = pred_missing = 0
    for t, p in zip(y_true, y_pred):
        if t is None:
            ref_missing += 1
            continue
        if p is None:
            pred_missing += 1
            continue
        if t not in (0, 1) or p not in (0, 1):
            raise ValueError(f"targets must be 0/1/None, got ({t!r}, {p!r})")
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return CountResult(BinaryCounts(tp, fp, tn, fn), len(y_true), ref_missing, pred_missing)


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def binary_rates(c: BinaryCounts) -> dict[str, float | None]:
    precision = _div(c.tp, c.tp + c.fp)
    recall = _div(c.tp, c.tp + c.fn)
    specificity = _div(c.tn, c.tn + c.fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    bal = None if (recall is None or specificity is None) else 0.5 * (recall + specificity)
    return {
        "accuracy": _div(c.tp + c.tn, c.n),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": _div(c.fp, c.fp + c.tn),
        "fnr": _div(c.fn, c.fn + c.tp),
        "f1": f1,
        "balanced_accuracy": bal,
        "n": c.n,
    }


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion; None when n == 0."""
    if n == 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


_METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "accuracy": lambda tp, fp, tn, fn: (tp + tn) / (tp + fp + tn + fn),
    "precision": lambda tp, fp, tn, fn: tp / (tp + fp),
    "recall": lambda tp, fp, tn, fn: tp / (tp + fn),
    "specificity": lambda tp, fp, tn, fn: tn / (tn + fp),
    "fpr": lambda tp, fp, tn, fn: fp / (fp + tn),
    "fnr": lambda tp, fp, tn, fn: fn / (fn + tp),
    "balanced_accuracy": lambda tp, fp, tn, fn: 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
}


def cluster_bootstrap_ci(
    cluster_counts: Sequence[BinaryCounts],
    metric: str,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    min_valid_fraction: float = 0.9,
) -> dict[str, float | int | None]:
    """Percentile CI from resampling whole clusters (episodes) with replacement.

    Returns {"lo", "hi", "n_valid", "n_boot"}; lo/hi are None if fewer than ``min_valid_fraction``
    of the resamples have a defined metric (e.g. no positives), in which case no interval is claimed.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"unknown metric {metric!r}")
    m = len(cluster_counts)
    if m == 0:
        return {"lo": None, "hi": None, "n_valid": 0, "n_boot": n_boot}
    arr = np.array([[c.tp, c.fp, c.tn, c.fn] for c in cluster_counts], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    sums = arr[idx].sum(axis=1)  # (n_boot, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = _METRIC_FNS[metric](sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3])
    vals = vals[np.isfinite(vals)]
    if len(vals) < min_valid_fraction * n_boot:
        return {"lo": None, "hi": None, "n_valid": int(len(vals)), "n_boot": n_boot}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "n_valid": int(len(vals)), "n_boot": n_boot}


def paired_cluster_bootstrap_diff(
    counts_ref: Sequence[BinaryCounts],
    counts_alt: Sequence[BinaryCounts],
    metric: str,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    min_valid_fraction: float = 0.9,
) -> dict[str, float | int | None]:
    """CI for metric(alt) - metric(ref) when both were evaluated on the SAME episodes (paired design).

    Each bootstrap draw resamples episode indices once and applies them to both conditions, so the shared
    episode-to-episode variation cancels. This is the right interval for comparing conditions on one audit set.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"unknown metric {metric!r}")
    if len(counts_ref) != len(counts_alt):
        raise ValueError("paired comparison needs the same episodes in the same order")
    m = len(counts_ref)
    if m == 0:
        return {"diff": None, "lo": None, "hi": None, "n_valid": 0, "n_boot": n_boot}
    a = np.array([[c.tp, c.fp, c.tn, c.fn] for c in counts_ref], dtype=float)
    b = np.array([[c.tp, c.fp, c.tn, c.fn] for c in counts_alt], dtype=float)
    fn = _METRIC_FNS[metric]
    with np.errstate(divide="ignore", invalid="ignore"):
        point = float(fn(*b.sum(axis=0)) - fn(*a.sum(axis=0)))
        idx = np.random.default_rng(seed).integers(0, m, size=(n_boot, m))
        sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
        d = fn(sb[:, 0], sb[:, 1], sb[:, 2], sb[:, 3]) - fn(sa[:, 0], sa[:, 1], sa[:, 2], sa[:, 3])
    d = d[np.isfinite(d)]
    if not math.isfinite(point) or len(d) < min_valid_fraction * n_boot:
        return {"diff": point if math.isfinite(point) else None, "lo": None, "hi": None, "n_valid": int(len(d)),
                "n_boot": n_boot}
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"diff": point, "lo": float(lo), "hi": float(hi), "n_valid": int(len(d)), "n_boot": n_boot}


def cohen_kappa(a: Sequence[Hashable | None], b: Sequence[Hashable | None]) -> dict[str, float | int | None]:
    """Cohen's kappa over pairs where both labels are present (categories compared as-is)."""
    if len(a) != len(b):
        raise ValueError("sequences must have equal length")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"kappa": None, "observed_agreement": None, "n": 0}
    po = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    kappa = None if pe == 1.0 else (po - pe) / (1 - pe)
    return {"kappa": kappa, "observed_agreement": po, "n": n}


def summarize_values(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Mean/SD/min/max over replicate values, ignoring None (count reported)."""
    v = [float(x) for x in values if x is not None]
    n_missing = len(values) - len(v)
    if not v:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0, "n_missing": n_missing}
    arr = np.array(v)
    sd = float(arr.std(ddof=1)) if len(v) > 1 else None
    return {
        "mean": float(arr.mean()),
        "sd": sd,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(v),
        "n_missing": n_missing,
    }
