#!/usr/bin/env python3
"""Seed-level power ESTIMATE for the online non-inferiority rule (DECISIONS D-30). Stdlib only.

Question: if B/C are TRULY exactly as good as A, how often does the pre-registered rule
    one-sided 95% Welch-t lower bound of (mean per-seed success of X - mean per-seed success of A) > -delta
conclude "non-inferior"?  (= power at zero true difference)

Every input is an ASSUMPTION, not a measurement. Replace with P3 pilot values:
  p        true success rate of every run's policy (same for both arms)
  seed_sd  between-seed SD of the true success rate (RL seed-to-seed variability)
  n_eval   evaluation episodes per run (binomial noise)
Model: run rate ~ Normal(p, seed_sd) clipped to [0, 1]; observed = Binomial(n_eval, rate) / n_eval.
"optimistic z" pretends the variance is known (ignores that it is estimated from ~3 seeds).
"""

from __future__ import annotations

import argparse
import math
import random
from statistics import NormalDist

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


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student-t CDF for real df > 0."""
    tail = 0.5 * betainc_reg(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Student-t quantile by bisection (display and tests only)."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch(x: list[float], a: list[float]) -> tuple[float, float, float]:
    """(mean(x) - mean(a), standard error, Welch-Satterthwaite df). df = inf when se == 0."""
    nx, na = len(x), len(a)
    if nx < 2 or na < 2:
        raise ValueError("need at least 2 completed runs per condition")
    mx, ma = sum(x) / nx, sum(a) / na
    sx = sum((v - mx) ** 2 for v in x) / (nx - 1) / nx
    sa = sum((v - ma) ** 2 for v in a) / (na - 1) / na
    se2 = sx + sa
    if se2 == 0.0:
        return mx - ma, 0.0, math.inf
    df = se2 * se2 / (sx * sx / (nx - 1) + sa * sa / (na - 1))
    return mx - ma, math.sqrt(se2), df


def welch_noninferior(x: list[float], a: list[float], delta: float, conf: float = 0.95) -> bool:
    """True iff the one-sided `conf` Welch lower bound of mean(x) - mean(a) is above -delta."""
    diff, se, df = welch(x, a)
    margin = diff + delta
    if se == 0.0:
        return margin > 0.0
    # lower bound > -delta  <=>  margin / se > t_conf(df)  <=>  t_cdf(margin / se) > conf
    return margin > 0.0 and t_cdf(margin / se, df) > conf


def simulate(p: float, seed_sd: float, seeds: int, n_eval: int, delta: float,
             sims: int, rng_seed: int = 0) -> tuple[float, float]:
    """(power with optimistic known-variance z, power with the pre-registered Welch-t rule)."""
    rng = random.Random(rng_seed)
    z95 = NormalDist().inv_cdf(0.95)
    se_known = math.sqrt(2.0 * (seed_sd ** 2 + p * (1.0 - p) / n_eval) / seeds)
    ok_z = ok_t = 0
    for _ in range(sims):
        arms = []
        for _arm in range(2):
            rates = []
            for _s in range(seeds):
                r = min(1.0, max(0.0, rng.gauss(p, seed_sd)))
                rates.append(sum(1 for _e in range(n_eval) if rng.random() < r) / n_eval)
            arms.append(rates)
        a, x = arms
        diff = sum(x) / seeds - sum(a) / seeds
        ok_z += (diff + delta) / se_known > z95
        ok_t += welch_noninferior(x, a, delta)
    return ok_z / sims, ok_t / sims


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="시드 수준 검정력 추정 (가정값으로 계산, 실측 아님)")
    ap.add_argument("--p", type=float, default=0.70, help="가정한 실제 성공률 (두 조건 같음)")
    ap.add_argument("--seed-sd", type=float, nargs="+", default=[0.10, 0.15], help="가정한 시드 간 표준편차")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5], help="조건당 시드(실행) 수")
    ap.add_argument("--n-eval", type=int, default=20, help="실행당 평가 에피소드 수")
    ap.add_argument("--delta", type=float, default=0.20, help="비열등성 마진")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--rng-seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(f"시드 수준 검정력 추정 (가정값, 실측 아님): p={a.p:.2f}, 실행당 평가 {a.n_eval}회, "
          f"δ={a.delta:.2f}, 시뮬레이션 {a.sims}회, 난수 시드 {a.rng_seed}")
    print("B/C가 실제로 A와 똑같을 때 '비열등' 결론이 나올 확률")
    print("| 조건당 시드 | 시드 간 SD | Welch t (사전 등록 규칙, D-30) | 분산을 안다고 가정 (낙관적) |")
    print("|---|---|---|---|")
    for r in a.seeds:
        for sd in a.seed_sd:
            pz, pt = simulate(a.p, sd, r, a.n_eval, a.delta, a.sims, a.rng_seed)
            print(f"| {r} | {sd:.2f} | {pt:.2f} | {pz:.2f} |")


if __name__ == "__main__":
    main()
