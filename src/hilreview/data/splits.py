"""Episode-level (or session-level) train/val/audit splitting with leakage checks.

Rules (DECISIONS.md D-06):
* Units of assignment are *groups* (default: episodes; optionally sessions). All frames/keyframes of a
  group land in exactly one split, so near-duplicate frames can never straddle train and audit.
* Ineligible episodes get split "excluded" and are never used by any condition.
* Stratification uses collection metadata only (never labels). Allocation across strata uses the
  largest-remainder method so the total audit size is exactly round_half_up(fraction * n_groups).
* The split depends only on (episodes, spec); it is independent of input order and of condition/seed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from hilreview.data.schema import Episode, Keyframe
from hilreview.utils.numeric import round_half_up
from hilreview.utils.rng import make_rng

SPLIT_NAMES = ("train", "val", "audit")
EXCLUDED = "excluded"


class LeakageError(RuntimeError):
    """Raised when split integrity is violated."""


@dataclass(frozen=True)
class SplitSpec:
    audit_fraction: float = 0.25
    val_fraction_of_pool: float = 0.2
    group_by: str = "episode_id"  # "episode_id" or "session_id"
    stratify_by: tuple[str, ...] = ("session_id", "collection_mode")
    seed: int = 20261017

    def __post_init__(self) -> None:
        if not 0.0 < self.audit_fraction < 1.0:
            raise ValueError("audit_fraction must be in (0, 1)")
        if not 0.0 <= self.val_fraction_of_pool < 1.0:
            raise ValueError("val_fraction_of_pool must be in [0, 1)")
        if self.group_by not in ("episode_id", "session_id"):
            raise ValueError("group_by must be 'episode_id' or 'session_id'")
        object.__setattr__(self, "stratify_by", tuple(self.stratify_by or ()))
        for key in self.stratify_by:
            if key not in ("session_id", "collection_mode"):
                raise ValueError(f"unsupported stratify_by key {key!r}")

    def to_dict(self) -> dict:
        return {
            "audit_fraction": self.audit_fraction,
            "val_fraction_of_pool": self.val_fraction_of_pool,
            "group_by": self.group_by,
            "stratify_by": list(self.stratify_by),
            "seed": self.seed,
        }


def _apportion(total: int, sizes: Mapping[tuple, int], rng: np.random.Generator) -> dict[tuple, int]:
    """Largest-remainder apportionment of ``total`` items across strata proportional to ``sizes``."""
    n = sum(sizes.values())
    if n == 0:
        return {k: 0 for k in sizes}
    keys = sorted(sizes)
    quotas = {k: total * sizes[k] / n for k in keys}
    alloc = {k: min(sizes[k], math.floor(quotas[k])) for k in keys}
    remaining = total - sum(alloc.values())
    tiebreak = {k: float(r) for k, r in zip(keys, rng.random(len(keys)))}
    order = sorted(keys, key=lambda k: (-(quotas[k] - math.floor(quotas[k])), tiebreak[k]))
    while remaining > 0:
        progressed = False
        for k in order:
            if remaining == 0:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def _pick(groups: Sequence[str], strata: Mapping[str, tuple], fraction: float, seed_parts: tuple) -> set[str]:
    """Select round_half_up(fraction * len(groups)) groups, stratified, deterministic given seed_parts."""
    total = round_half_up(fraction * len(groups))
    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for g in sorted(groups):
        by_stratum[strata[g]].append(g)
    rng = make_rng(*seed_parts, "apportion")
    alloc = _apportion(total, {s: len(v) for s, v in by_stratum.items()}, rng)
    chosen: set[str] = set()
    for s in sorted(by_stratum):
        members = by_stratum[s]
        perm = make_rng(*seed_parts, "stratum", repr(s)).permutation(len(members))
        chosen.update(members[i] for i in perm[: alloc[s]])
    return chosen


def assign_splits(episodes: Sequence[Episode], spec: SplitSpec) -> dict[str, str]:
    """Map episode_id -> one of 'train' | 'val' | 'audit' | 'excluded'."""
    ids = [e.episode_id for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode_id in input")

    group_of: dict[str, str] = {}
    members: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        if not e.eligible:
            continue
        g = getattr(e, spec.group_by)
        group_of[e.episode_id] = g
        members[g].append(e)

    strata: dict[str, tuple] = {}
    for g, eps in members.items():
        values = {tuple(getattr(e, key) for key in spec.stratify_by) for e in eps}
        if len(values) > 1:
            raise ValueError(
                f"stratify_by={spec.stratify_by} is not constant within group {g!r} (group_by={spec.group_by}); "
                "use group_by='episode_id' or drop the varying stratification key"
            )
        strata[g] = values.pop()

    groups = sorted(members)
    audit_groups = _pick(groups, strata, spec.audit_fraction, (spec.seed, "audit"))
    pool_groups = [g for g in groups if g not in audit_groups]
    val_groups = _pick(pool_groups, strata, spec.val_fraction_of_pool, (spec.seed, "val"))

    out: dict[str, str] = {}
    for e in episodes:
        if not e.eligible:
            out[e.episode_id] = EXCLUDED
            continue
        g = group_of[e.episode_id]
        out[e.episode_id] = "audit" if g in audit_groups else ("val" if g in val_groups else "train")
    return out


def check_split_integrity(
    episodes: Sequence[Episode],
    split_of: Mapping[str, str],
    keyframes_by_episode: Mapping[str, Sequence[Keyframe]] | None = None,
    group_by: str = "episode_id",
) -> dict[str, int]:
    """Raise LeakageError on any violation; return split counts on success.

    Checks: every episode assigned exactly once with a valid split name; ineligible -> excluded;
    no group spans splits; no non-empty ``source`` appears in two splits (catches duplicated
    episodes under different IDs); keyframe sample_ids are globally unique and belong to known episodes.
    """
    valid = set(SPLIT_NAMES) | {EXCLUDED}
    ep_by_id = {e.episode_id: e for e in episodes}
    if set(split_of) != set(ep_by_id):
        missing = set(ep_by_id) - set(split_of)
        extra = set(split_of) - set(ep_by_id)
        raise LeakageError(f"split map mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    for eid, s in split_of.items():
        if s not in valid:
            raise LeakageError(f"episode {eid!r} has invalid split {s!r}")
        if not ep_by_id[eid].eligible and s != EXCLUDED:
            raise LeakageError(f"ineligible episode {eid!r} assigned to {s!r}")
        if ep_by_id[eid].eligible and s == EXCLUDED:
            raise LeakageError(f"eligible episode {eid!r} marked excluded")

    for key in {group_by, "episode_id"}:
        seen: dict[str, str] = {}
        for e in episodes:
            s = split_of[e.episode_id]
            if s == EXCLUDED:
                continue
            g = getattr(e, key)
            if g in seen and seen[g] != s:
                raise LeakageError(f"{key}={g!r} appears in both {seen[g]!r} and {s!r}")
            seen[g] = s

    src_seen: dict[str, str] = {}
    for e in episodes:
        s = split_of[e.episode_id]
        if s == EXCLUDED or not e.source:
            continue
        if e.source in src_seen and src_seen[e.source] != s:
            raise LeakageError(f"source {e.source!r} appears in both {src_seen[e.source]!r} and {s!r}")
        src_seen[e.source] = s

    if keyframes_by_episode is not None:
        seen_ids: set[str] = set()
        for eid, kfs in keyframes_by_episode.items():
            if eid not in ep_by_id:
                raise LeakageError(f"keyframes reference unknown episode {eid!r}")
            for kf in kfs:
                if kf.episode_id != eid:
                    raise LeakageError(f"keyframe {kf.sample_id!r} filed under wrong episode {eid!r}")
                if kf.sample_id in seen_ids:
                    raise LeakageError(f"duplicate sample_id {kf.sample_id!r}")
                seen_ids.add(kf.sample_id)

    counts = {s: 0 for s in (*SPLIT_NAMES, EXCLUDED)}
    for s in split_of.values():
        counts[s] += 1
    return counts
