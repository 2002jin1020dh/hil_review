"""Review-selection policies (DECISIONS.md D-04, D-05).

A policy chooses WHICH eligible pool episodes a human reviews, given a budget ``k``.
By construction, a policy only receives VLM outputs (labels, scores), episode IDs and an RNG;
it never receives human labels or audit data, so selection cannot leak ground truth.

Budget rule: k = round_half_up(fraction * N_eligible_pool_episodes), clipped to [0, N].
Ties are broken by a seeded random permutation, and inputs are sorted by episode_id first,
so selections are independent of input order and reproducible from (seed, replicate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hilreview.data.schema import Label
from hilreview.utils.numeric import round_half_up

POLICIES = ("none", "full", "random", "low_confidence", "fake_success_priority")


def review_budget(fraction: float, n_eligible: int) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if n_eligible < 0:
        raise ValueError("n_eligible must be >= 0")
    return min(n_eligible, round_half_up(fraction * n_eligible))


@dataclass(frozen=True)
class EpisodeVLMView:
    """Everything a selection policy is allowed to know about one episode."""

    episode_id: str
    labels: tuple[Label | None, ...]
    p_success: tuple[float | None, ...]


def keyframe_uncertainty(label: Label | None, p: float | None) -> float:
    """1 - |2p - 1|: 1 at p = 0.5, 0 at p in {0, 1}. A keyframe without a usable label
    (VLM error, 'unsure', or no score) counts as 1. Used for ranking only; not a calibrated quantity."""
    if label is None or label is Label.AMBIGUOUS or p is None:
        return 1.0
    return 1.0 - abs(2.0 * p - 1.0)


def episode_uncertainty(view: EpisodeVLMView) -> float:
    """Sum of keyframe uncertainties.

    If p_success were calibrated, min(p, 1 - p) = u / 2 would be the error probability of a keyframe, so the sum
    is proportional to the expected number of label errors that reviewing this episode would fix. (A max over
    keyframes saturates: one 'unsure' keyframe would tie most episodes at the maximum.)
    """
    return float(sum(keyframe_uncertainty(lab, p) for lab, p in zip(view.labels, view.p_success)))


def fake_success_key(view: EpisodeVLMView) -> tuple[bool, float]:
    """(VLM claims success somewhere, expected number of fake-success labels = sum of (1 - p) over claimed keyframes).

    If p were calibrated, 1 - p_i would be the probability that a VLM 'success' keyframe is truly a failure, so the
    sum is the expected number of fake-success labels that reviewing this episode would remove. A missing p counts
    as 0.5. (v1 ranked by the MEAN p of claimed keyframes; in the synthetic simulator that was worse than random at
    reducing false-positive labels, because long true-success episodes containing hover/boundary false positives
    were ranked last. See DECISIONS.md D-17.)
    """
    risks = [1.0 - (0.5 if p is None else p) for lab, p in zip(view.labels, view.p_success) if lab is Label.SUCCESS]
    if not risks:
        return (False, 0.0)
    return (True, float(sum(risks)))


def select_episodes(policy: str, views: Sequence[EpisodeVLMView], k: int, rng: np.random.Generator) -> list[str]:
    """Return selected episode IDs in priority order. ``none``/``full`` ignore k (validated elsewhere)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    ordered = sorted(views, key=lambda v: v.episode_id)
    ids = [v.episode_id for v in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode_id in views")
    if policy == "none":
        return []
    if policy == "full":
        return ids
    if not 0 <= k <= len(ids):
        raise ValueError(f"budget k={k} outside [0, {len(ids)}]")
    tiebreak = rng.permutation(len(ids))  # consumed identically by every policy -> comparable RNG use
    if policy == "random":
        return [ids[i] for i in tiebreak[:k]]
    tb = {eid: int(t) for eid, t in zip(ids, tiebreak)}
    if policy == "low_confidence":
        ranked = sorted(ordered, key=lambda v: (-episode_uncertainty(v), tb[v.episode_id]))
        return [v.episode_id for v in ranked[:k]]
    # fake_success_priority: VLM-claimed successes first, most expected fake-success labels first;
    # then the remaining episodes by low confidence
    claimed = [v for v in ordered if fake_success_key(v)[0]]
    rest = [v for v in ordered if not fake_success_key(v)[0]]
    claimed.sort(key=lambda v: (-fake_success_key(v)[1], tb[v.episode_id]))
    rest.sort(key=lambda v: (-episode_uncertainty(v), tb[v.episode_id]))
    return [v.episode_id for v in (claimed + rest)[:k]]
