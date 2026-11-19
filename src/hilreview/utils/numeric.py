"""Small numeric helpers shared by protocol code (rounding rules must be identical everywhere)."""

from __future__ import annotations

import math


def round_half_up(x: float) -> int:
    """Round to nearest integer, halves away from zero for x >= 0 (2.5 -> 3, 18.75 -> 19).

    Python's built-in round() uses banker's rounding (2.5 -> 2), which would make the
    review-budget definition depend on an implementation detail; the protocol fixes half-up.
    """
    if x < 0:
        raise ValueError("round_half_up is only defined here for x >= 0")
    return int(math.floor(x + 0.5))
