"""D-30 online non-inferiority: per-seed eval success rates, "X - A", one-sided 95% Welch t bounds. Stdlib only.

Rule (pre-registered, DECISIONS.md D-30):
  diff = mean(per-seed success of X) - mean(per-seed success of A)
  se   = sqrt(s_x^2/n_x + s_a^2/n_a),  df = Welch-Satterthwaite
  lower = diff - t_{conf}(df) * se      (one-sided conf lower bound)
  upper = diff + t_{conf}(df) * se      (one-sided conf upper bound)
  lower > -delta  -> "noninferior";  upper < -delta -> "inferior";  otherwise "inconclusive".
Fewer than 2 runs in either arm -> "insufficient_runs" (no variance estimate).
If se == 0 (all seeds identical in both arms) the bounds collapse to diff.

The same Welch computation lives in scripts/power_seeds.py (planning script, kept standalone);
tests/test_noninferiority.py checks the two agree.
"""

from __future__ import annotations

import math
from typing import Sequence

DECISIONS = ("noninferior", "inferior", "inconclusive", "insufficient_runs")
_TINY = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        step = d * c
        h *= step
        if abs(step - 1.0) < 3e-15:
            break
    return h


def _betainc_reg(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student-t CDF for real df > 0 (df = inf -> standard normal)."""
    if math.isinf(df):
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    tail = 0.5 * _betainc_reg(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Student-t quantile (bisection on t_cdf; accurate to ~1e-10)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch(x: Sequence[float], a: Sequence[float]) -> tuple[float, float, float]:
    """(mean(x) - mean(a), standard error, Welch-Satterthwaite df). df = inf when se == 0."""
    nx, na = len(x), len(a)
    if nx < 2 or na < 2:
        raise ValueError("need at least 2 runs per condition")
    mx, ma = sum(x) / nx, sum(a) / na
    vx = sum((v - mx) ** 2 for v in x) / (nx - 1) / nx
    va = sum((v - ma) ** 2 for v in a) / (na - 1) / na
    se2 = vx + va
    if se2 == 0.0:
        return mx - ma, 0.0, math.inf
    # a zero-variance arm contributes 0 to the denominator; the other arm then carries all df
    den = (vx * vx / (nx - 1) if vx else 0.0) + (va * va / (na - 1) if va else 0.0)
    return mx - ma, math.sqrt(se2), se2 * se2 / den


def noninferiority(x: Sequence[float], a: Sequence[float], delta: float = 0.20, conf: float = 0.95) -> dict:
    """Apply the D-30 rule to per-seed success rates of condition X and reference A."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.5 < conf < 1.0:
        raise ValueError("conf must be in (0.5, 1)")
    for v in list(x) + list(a):
        if v is None or not 0.0 <= v <= 1.0:
            raise ValueError(f"per-seed success rates must be in [0, 1], got {v!r}")
    out = {"n_x": len(x), "n_a": len(a), "delta": delta, "conf": conf,
           "mean_x": (sum(x) / len(x)) if x else None, "mean_a": (sum(a) / len(a)) if a else None}
    if len(x) < 2 or len(a) < 2:
        return out | {"decision": "insufficient_runs", "diff": None, "se": None, "df": None,
                      "t_crit": None, "lower": None, "upper": None}
    diff, se, df = welch(x, a)
    tq = t_ppf(conf, df) if se > 0 else 0.0
    lower, upper = diff - tq * se, diff + tq * se
    if lower > -delta:
        decision = "noninferior"
    elif upper < -delta:
        decision = "inferior"
    else:
        decision = "inconclusive"
    return out | {"decision": decision, "diff": diff, "se": se, "df": df, "t_crit": tq,
                  "lower": lower, "upper": upper}
