"""Deterministic seeding helpers.

Python's built-in ``hash()`` is randomized per process, so it must never be used for seeding.
``derive_seed`` hashes its arguments with SHA-256, which is stable across runs and machines.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(*parts: object) -> int:
    """Return a stable 63-bit integer seed derived from ``parts`` (e.g. base seed, purpose, replicate)."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def make_rng(*parts: object) -> np.random.Generator:
    """numpy Generator seeded with ``derive_seed(*parts)``."""
    return np.random.default_rng(derive_seed(*parts))
