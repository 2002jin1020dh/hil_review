#!/usr/bin/env python3
"""Formula-based planning ESTIMATES (not measurements). Stdlib only, so it runs anywhere.

1) Non-inferiority sample size for two success proportions (normal approximation, equal true rates):
       n_per_arm = (z_{1-alpha} + z_{1-beta})^2 * 2 p (1 - p) / delta^2
   with one-sided alpha. Eval episodes clustered within runs inflate this by the design effect
       DE = 1 + (m - 1) * ICC     (m = eval episodes per run, ICC = intra-run correlation; both assumed).
2) 95% Wilson half-width for a single condition's success rate with n eval episodes.
3) Human-time scenarios from assumed seconds per episode (replace with P2 pilot MEASUREMENTS).
Re-run with your own assumptions; every input below is an assumption.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def n_noninferiority(p: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return math.ceil(z * z * 2 * p * (1 - p) / (delta * delta))


def wilson_half_width(p: float, n: int, z: float = 1.959963984540054) -> float:
    denom = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom


def main() -> None:
    print("== 1) Non-inferiority: eval episodes PER CONDITION (one-sided alpha=0.05, power=0.80) ==")
    print("   independent episodes; multiply by the design effect for clustered episodes")
    print(f"{'p (true success)':>18} | " + " | ".join(f"delta={d:.2f}" for d in (0.10, 0.15, 0.20)))
    for p in (0.7, 0.8, 0.9):
        print(f"{p:>18.2f} | " + " | ".join(f"{n_noninferiority(p, d):>10d}" for d in (0.10, 0.15, 0.20)))
    print("\n   design effect DE = 1 + (m-1)*ICC")
    for m in (10, 20):
        print(f"   m={m:2d} eval episodes/run: " + ", ".join(f"ICC={icc:.2f} -> DE={1 + (m - 1) * icc:.2f}"
                                                         for icc in (0.05, 0.10, 0.20)))
    n0 = n_noninferiority(0.8, 0.15)
    print(f"   example: p=0.8, delta=0.15 -> {n0} independent episodes; with m=20, ICC=0.10 -> "
          f"{math.ceil(n0 * (1 + 19 * 0.10))} episodes per condition")

    print("\n== 2) 95% Wilson half-width of one condition's success rate (p = 0.8) ==")
    for n in (20, 40, 60, 100, 200):
        print(f"   n={n:3d} eval episodes -> +/- {wilson_half_width(0.8, n):.3f}")

    print("\n== 3) Human-time scenarios (ASSUMED seconds per episode; replace with pilot P2 measurements) ==")
    pool, audit, repeat = 90, 30, 9
    for s_assist, s_blind in ((10, 20), (20, 40), (40, 80)):
        full = pool * s_assist / 60
        print(f"   assisted {s_assist:2d} s/ep, blind {s_blind:2d} s/ep: A(100%)={full:5.1f} min, "
              f"B(25%)~{0.25 * full:5.1f} min, C=0 min | audit+repeat (all conditions share it)="
              f"{(audit + repeat) * s_blind / 60:5.1f} min")

    print("\n== 4) VLM labeling compute (ASSUMED latency; measure in pilot P1) ==")
    n_kf = 120 * (20 * 10 // 5 + 1)
    for lat in (0.3, 1.0, 2.0):
        print(f"   {n_kf} keyframes x {lat:.1f} s = {n_kf * lat / 3600:4.1f} GPU-hours")


if __name__ == "__main__":
    main()
