"""Assemble the final training labels of one experimental condition (DECISIONS.md D-05, D-07).

Inputs are fixed across conditions: the eligible pool episodes, their keyframes, ONE set of VLM suggestions
and ONE full assisted human review log ("review once, subsample many"). A condition only decides which
episodes take the human labels; all others take the VLM label (or no label if the VLM failed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from hilreview.data.schema import Keyframe, Label, ReviewRecord, VLMSuggestion, label_to_target
from hilreview.review.policies import EpisodeVLMView, review_budget, select_episodes


class MissingReviewError(RuntimeError):
    """A selected episode has no (complete) human review record."""


BUDGET_UNITS = ("episodes", "seconds")


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    policy: str  # none | full | random | low_confidence | fake_success_priority
    fraction: float
    # "episodes": k = round_half_up(fraction * N) episodes (protocol default, DECISIONS.md D-04).
    # "seconds": review in priority order while time remains; budget = fraction * total (idle-capped) review time of
    #            the full pool review. An episode is started only if budget remains, so the last one may overshoot
    #            (as in real life: you finish the episode you are reviewing). Uses review TIMES, never labels.
    budget_unit: str = "episodes"

    def __post_init__(self) -> None:
        if self.budget_unit not in BUDGET_UNITS:
            raise ValueError(f"{self.name}: budget_unit must be one of {BUDGET_UNITS}")
        if self.policy == "none" and self.fraction != 0.0:
            raise ValueError(f"{self.name}: policy 'none' requires fraction 0")
        if self.policy == "full" and self.fraction != 1.0:
            raise ValueError(f"{self.name}: policy 'full' requires fraction 1")
        if self.policy not in ("none", "full") and not 0.0 < self.fraction < 1.0:
            raise ValueError(f"{self.name}: partial policies require 0 < fraction < 1")
        if self.budget_unit == "seconds" and self.policy in ("none", "full"):
            raise ValueError(f"{self.name}: a time budget only makes sense for partial policies")


@dataclass
class ConditionLabels:
    spec: ConditionSpec
    k: int
    n_eligible_pool_episodes: int
    selected_episodes: list[str]  # human-reviewed episodes, priority order
    labels: dict[str, Label | None]  # sample_id -> final label
    sources: dict[str, str]  # sample_id -> "human" | "vlm" | "none" | "auto_invalid"
    meta: dict = field(default_factory=dict)

    def targets(self, sample_ids: Sequence[str]) -> list[int | None]:
        return [label_to_target(self.labels.get(s)) for s in sample_ids]

    def label_counts(self) -> dict[str, int]:
        c = {"n_total": len(self.labels), "success": 0, "failure": 0, "ambiguous": 0, "invalid": 0, "no_label": 0}
        for lab in self.labels.values():
            if lab is None:
                c["no_label"] += 1
            else:
                c[lab.value] += 1
        c["n_trainable"] = c["success"] + c["failure"]
        c["n_excluded"] = c["n_total"] - c["n_trainable"]
        c["human_labeled"] = sum(1 for s in self.sources.values() if s == "human")
        c["vlm_labeled"] = sum(1 for s in self.sources.values() if s == "vlm")
        return c


def episode_views(
    episode_ids: Sequence[str],
    keyframes_by_episode: Mapping[str, Sequence[Keyframe]],
    vlm_by_sample: Mapping[str, VLMSuggestion],
) -> list[EpisodeVLMView]:
    views = []
    for eid in episode_ids:
        labels, ps = [], []
        for kf in keyframes_by_episode[eid]:
            if not kf.auto_valid:
                continue
            s = vlm_by_sample.get(kf.sample_id)
            labels.append(None if s is None else s.label)
            ps.append(None if s is None else s.p_success)
        views.append(EpisodeVLMView(eid, tuple(labels), tuple(ps)))
    return views


def capped_review_times(pool: Sequence[str], pool_reviews: Mapping[str, ReviewRecord],
                        idle_cap_s: float) -> dict[str, float] | None:
    """Idle-capped review time per pool episode, or None if any pool episode lacks a time."""
    times = {}
    for eid in pool:
        rec = pool_reviews.get(eid)
        if rec is None or rec.active_time_s is None:
            return None
        times[eid] = min(float(rec.active_time_s), idle_cap_s)
    return times


def build_condition_labels(
    spec: ConditionSpec,
    pool_episode_ids: Sequence[str],
    keyframes_by_episode: Mapping[str, Sequence[Keyframe]],
    vlm_by_sample: Mapping[str, VLMSuggestion],
    pool_reviews: Mapping[str, ReviewRecord],
    rng: np.random.Generator,
    idle_cap_s: float = 120.0,
) -> ConditionLabels:
    pool = sorted(pool_episode_ids)
    views = episode_views(pool, keyframes_by_episode, vlm_by_sample)
    times = capped_review_times(pool, pool_reviews, idle_cap_s)
    meta: dict = {"budget_unit": spec.budget_unit}
    if spec.budget_unit == "seconds":
        if times is None:
            raise MissingReviewError(f"{spec.name}: a time budget needs a review time for every pool episode")
        total = math.fsum(times.values())
        budget_s = spec.fraction * total
        order = select_episodes(spec.policy, views, len(pool), rng)  # full priority order, same RNG use
        selected, used = [], 0.0
        for eid in order:
            if used >= budget_s:
                break
            selected.append(eid)
            used += times[eid]
        k = len(selected)
        meta.update({"time_budget_s": budget_s, "time_total_s": total})
    else:
        k = review_budget(spec.fraction, len(pool))
        selected = select_episodes(spec.policy, views, k, rng)
        if spec.policy == "full":
            k = len(selected)
    if times is not None:
        used = math.fsum(times[e] for e in selected)
        total = math.fsum(times.values())
        meta.update({"time_used_s": used, "realized_time_fraction": (used / total) if total > 0 else None})
    selected_set = set(selected)

    labels: dict[str, Label | None] = {}
    sources: dict[str, str] = {}
    for eid in pool:
        rec = pool_reviews.get(eid) if eid in selected_set else None
        if eid in selected_set:
            if rec is None:
                raise MissingReviewError(f"episode {eid!r} selected for review but has no pool_review record")
            if rec.purpose != "pool_review":
                raise MissingReviewError(f"episode {eid!r}: expected a pool_review record, got {rec.purpose!r}")
        for kf in keyframes_by_episode[eid]:
            sid = kf.sample_id
            if not kf.auto_valid:  # identical treatment in every condition
                labels[sid], sources[sid] = Label.INVALID, "auto_invalid"
            elif rec is not None:
                if sid not in rec.labels:
                    raise MissingReviewError(f"review of {eid!r} lacks keyframe {sid!r}")
                labels[sid], sources[sid] = rec.labels[sid], "human"
            else:
                s = vlm_by_sample.get(sid)
                if s is None or s.label is None:
                    labels[sid], sources[sid] = None, "none"
                else:
                    labels[sid], sources[sid] = s.label, "vlm"
    return ConditionLabels(spec, k, len(pool), selected, labels, sources, meta)
