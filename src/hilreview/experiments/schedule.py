"""Blinded, block-randomized schedule for real-robot runs (DECISIONS.md D-13).

Each block contains every condition once in random order (one block per replicate), so slow drifts
(lighting, servo wear, operator learning) are spread evenly. Run IDs are opaque; the mapping
run_id -> condition is written to a SEPARATE key file that the human operator should not open until all
runs and success judgments are finished.
"""

from __future__ import annotations

from typing import Sequence

from hilreview.utils.rng import make_rng


def make_blinded_schedule(conditions: Sequence[str], n_replicates: int, seed: int) -> tuple[list[dict], list[dict]]:
    if len(set(conditions)) != len(conditions) or not conditions:
        raise ValueError("conditions must be non-empty and unique")
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    rng = make_rng(seed, "blinded_schedule")
    schedule, key, used = [], [], set()
    order = 0
    for block in range(n_replicates):
        for pos in rng.permutation(len(conditions)):
            while True:
                run_id = f"run-{int(rng.integers(0, 16**8)):08x}"
                if run_id not in used:
                    used.add(run_id)
                    break
            schedule.append({"order": order, "block": block, "run_id": run_id})
            key.append({"run_id": run_id, "condition": conditions[int(pos)], "replicate": block})
            order += 1
    return schedule, key
