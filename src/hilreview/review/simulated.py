"""SIMULATED human reviewer (synthetic data only).

Models two sessions from the protocol:
* assisted pool review (VLM suggestions shown): humans fix most VLM errors, but sometimes accept a wrong
  suggestion (automation bias) and make their own mistakes near state changes;
* blind audit labeling (no VLM shown).
All numbers in SimulatedReviewerProfile are INVENTED PLACEHOLDERS for pipeline testing. They are not
estimates of real human accuracy or speed. Every record has timing_source="simulated".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from hilreview.data.schema import (
    STATE_SUCCESS,
    DataKind,
    Keyframe,
    Label,
    ReviewRecord,
    SyntheticKeyframeTruth,
    VLMSuggestion,
)
from hilreview.utils.rng import make_rng


@dataclass(frozen=True)
class SimulatedReviewerProfile:
    miss_vlm_error: float = 0.10  # P(accepting a wrong, definite VLM label) in assisted review
    assisted_base_error: float = 0.005
    assisted_boundary_error: float = 0.08
    blind_base_error: float = 0.01
    blind_boundary_error: float = 0.08
    occluded_ambiguous_prob: float = 0.7
    assisted_base_s: float = 6.0
    assisted_per_keyframe_s: float = 0.25
    per_correction_s: float = 1.5
    blind_base_s: float = 8.0
    blind_per_keyframe_s: float = 0.6
    time_lognormal_sigma: float = 0.35
    idle_prob: float = 0.01  # reviewer occasionally walks away -> long outlier time
    idle_extra_s: float = 300.0


def _flip(label: Label) -> Label:
    return Label.FAILURE if label is Label.SUCCESS else Label.SUCCESS


class SimulatedReviewer:
    def __init__(
        self,
        truth: Mapping[str, SyntheticKeyframeTruth],
        profile: SimulatedReviewerProfile,
        seed: int,
        reviewer_id: str = "sim-reviewer-1",
    ):
        self.truth = truth
        self.p = profile
        self.seed = seed
        self.reviewer_id = reviewer_id

    def _time(self, rng, base: float, per_kf: float, n_kf: int, n_corr: int) -> float:
        t = (base + per_kf * n_kf + self.p.per_correction_s * n_corr) * float(
            rng.lognormal(0.0, self.p.time_lognormal_sigma)
        )
        if rng.random() < self.p.idle_prob:
            t += self.p.idle_extra_s
        return round(t, 3)

    def _true_label(self, sid: str) -> Label:
        return Label.SUCCESS if self.truth[sid].state == STATE_SUCCESS else Label.FAILURE

    def review_pool(
        self,
        episode_order: Sequence[str],
        keyframes_by_episode: Mapping[str, Sequence[Keyframe]],
        vlm_by_sample: Mapping[str, VLMSuggestion],
    ) -> dict[str, ReviewRecord]:
        """Assisted review of every pool episode, in the given (randomized) session order."""
        out = {}
        p = self.p
        for order, eid in enumerate(episode_order):
            rng = make_rng(self.seed, "pool_review", eid)
            labels, n_corr = {}, 0
            for kf in keyframes_by_episode[eid]:
                sid = kf.sample_id
                t = self.truth[sid]
                s = vlm_by_sample.get(sid)
                v = None if s is None else s.label
                if not kf.auto_valid:
                    lab = Label.INVALID
                elif t.occluded and rng.random() < p.occluded_ambiguous_prob:
                    lab = Label.AMBIGUOUS
                else:
                    true = self._true_label(sid)
                    if v in (Label.SUCCESS, Label.FAILURE) and v is not true and rng.random() < p.miss_vlm_error:
                        lab = v  # automation bias: wrong suggestion accepted
                    else:
                        err = p.assisted_boundary_error if t.boundary else p.assisted_base_error
                        lab = _flip(true) if rng.random() < err else true
                if lab is not v:
                    n_corr += 1
                labels[sid] = lab
            n_kf = len(keyframes_by_episode[eid])
            out[eid] = ReviewRecord(
                eid, self.reviewer_id, "pool_review", True, labels,
                self._time(rng, p.assisted_base_s, p.assisted_per_keyframe_s, n_kf, n_corr),
                "simulated", DataKind.SYNTHETIC, review_order=order, session_id="sim-pool-session",
            )
        return out

    def label_blind(
        self,
        episode_order: Sequence[str],
        keyframes_by_episode: Mapping[str, Sequence[Keyframe]],
        purpose: str = "audit",
    ) -> dict[str, ReviewRecord]:
        """Blind labeling (no VLM shown), used for the audit set and its repeat pass."""
        out = {}
        p = self.p
        for order, eid in enumerate(episode_order):
            rng = make_rng(self.seed, purpose, eid)
            labels = {}
            for kf in keyframes_by_episode[eid]:
                sid = kf.sample_id
                t = self.truth[sid]
                if not kf.auto_valid:
                    lab = Label.INVALID
                elif t.occluded and rng.random() < p.occluded_ambiguous_prob:
                    lab = Label.AMBIGUOUS
                else:
                    true = self._true_label(sid)
                    err = p.blind_boundary_error if t.boundary else p.blind_base_error
                    lab = _flip(true) if rng.random() < err else true
                labels[sid] = lab
            n_kf = len(keyframes_by_episode[eid])
            out[eid] = ReviewRecord(
                eid, self.reviewer_id, purpose, False, labels,
                self._time(rng, p.blind_base_s, p.blind_per_keyframe_s, n_kf, 0),
                "simulated", DataKind.SYNTHETIC, review_order=order, session_id=f"sim-{purpose}-session",
            )
        return out
