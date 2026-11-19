"""Offline replay of reward-classifier termination over recorded episodes (DECISIONS.md D-10).

LeRobot 0.6.1 ``RewardClassifierProcessorStep`` gives reward=1 and (with terminate_on_success) ends the
episode the FIRST time ``p > success_threshold`` on a single frame (strict '>'; checked in the source).
So the per-frame false-positive rate understates the damage: one false-positive frame ends an episode
with a false reward. This module replays per-keyframe scores against per-keyframe true states
("S" success, "F" failure, "U" unknown/ambiguous) and classifies the first trigger.

Outcome categories (per episode):
  true_trigger       trigger at a keyframe whose true state is S (within ``tolerance`` keyframes)
  early_trigger      trigger while state is F, but the episode reaches S later  -> premature false success
  false_trigger      trigger while state is F and no S follows                  -> false success
  ambiguous_trigger  trigger at a U keyframe with no S within tolerance (excluded from rates, counted)
  missed_success     no trigger although the episode contains S
  true_no_trigger    no trigger and no S

False-success termination (FST) = early_trigger + false_trigger.

This is a PROXY: recorded audit episodes (teleop/rollouts) differ from the online RL policy's
state distribution, and keyframe resolution (e.g. 2 Hz) is coarser than the runtime control rate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from hilreview.data.schema import STATE_FAILURE, STATE_SUCCESS, STATE_UNKNOWN
from hilreview.evaluation.metrics import wilson_interval

OUTCOMES = (
    "true_trigger",
    "early_trigger",
    "false_trigger",
    "ambiguous_trigger",
    "missed_success",
    "true_no_trigger",
)


@dataclass(frozen=True)
class EpisodeReplay:
    episode_id: str
    outcome: str
    trigger_index: int | None
    first_success_index: int | None
    n_keyframes: int


def first_trigger(scores: Sequence[float | None], threshold: float, debounce_k: int = 1) -> int | None:
    """Index of the first keyframe at which ``debounce_k`` consecutive scores are > threshold.

    debounce_k=1 reproduces LeRobot's behaviour. A None score (e.g. missing frame) resets the streak.
    """
    if debounce_k < 1:
        raise ValueError("debounce_k must be >= 1")
    streak = 0
    for i, s in enumerate(scores):
        if s is not None and s > threshold:
            streak += 1
            if streak >= debounce_k:
                return i
        else:
            streak = 0
    return None


def classify_episode(
    episode_id: str,
    scores: Sequence[float | None],
    states: Sequence[str],
    threshold: float,
    debounce_k: int = 1,
    tolerance: int = 0,
) -> EpisodeReplay:
    if len(scores) != len(states):
        raise ValueError(f"{episode_id}: scores and states differ in length")
    for s in states:
        if s not in (STATE_SUCCESS, STATE_FAILURE, STATE_UNKNOWN):
            raise ValueError(f"{episode_id}: invalid state {s!r}")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    n = len(states)
    success_idx = [i for i, s in enumerate(states) if s == STATE_SUCCESS]
    first_s = success_idx[0] if success_idx else None
    t = first_trigger(scores, threshold, debounce_k)
    if t is None:
        outcome = "missed_success" if success_idx else "true_no_trigger"
    else:
        lo, hi = max(0, t - tolerance), min(n - 1, t + tolerance)
        if any(states[j] == STATE_SUCCESS for j in range(lo, hi + 1)):
            outcome = "true_trigger"
        elif states[t] == STATE_UNKNOWN:
            outcome = "ambiguous_trigger"
        elif any(i > t for i in success_idx):
            outcome = "early_trigger"
        else:
            outcome = "false_trigger"
    return EpisodeReplay(episode_id, outcome, t, first_s, n)


def summarize_replays(replays: Sequence[EpisodeReplay]) -> dict:
    """Counts and rates. Denominators are explicit; undefined rates are None."""
    c = Counter(r.outcome for r in replays)
    n = len(replays)
    n_amb = c["ambiguous_trigger"]
    n_eval = n - n_amb  # episodes with a decidable outcome
    fst = c["early_trigger"] + c["false_trigger"]
    n_triggered_decidable = c["true_trigger"] + fst
    n_with_success = c["true_trigger"] + c["missed_success"] + c["early_trigger"]
    delays = [r.trigger_index - r.first_success_index for r in replays
              if r.outcome == "true_trigger" and r.first_success_index is not None and r.trigger_index is not None]
    ci = wilson_interval(fst, n_eval)
    out = {
        "n_episodes": n,
        "n_ambiguous_trigger_excluded": n_amb,
        "counts": {k: c.get(k, 0) for k in OUTCOMES},
        "fst_count": fst,
        # FST per decidable episode: chance an episode ends with a false reward.
        "fst_rate": None if n_eval == 0 else fst / n_eval,
        "fst_rate_ci95": None if ci is None else {"lo": ci[0], "hi": ci[1]},
        # Share of (decidable) terminations that were false.
        "fst_share_of_triggers": None if n_triggered_decidable == 0 else fst / n_triggered_decidable,
        # Missed successes among episodes that do reach success (and were not cut early).
        "missed_success_rate": None if n_with_success == 0 else c["missed_success"] / n_with_success,
        "mean_trigger_delay_keyframes": None if not delays else sum(delays) / len(delays),
    }
    return out


def replay_all(
    scores_by_episode: Mapping[str, Sequence[float | None]],
    states_by_episode: Mapping[str, Sequence[str]],
    threshold: float,
    debounce_k: int = 1,
    tolerance: int = 0,
) -> list[EpisodeReplay]:
    missing = set(states_by_episode) - set(scores_by_episode)
    if missing:
        raise ValueError(f"no scores for audit episodes: {sorted(missing)[:5]}")
    return [
        classify_episode(eid, scores_by_episode[eid], states_by_episode[eid], threshold, debounce_k, tolerance)
        for eid in sorted(states_by_episode)
    ]


def fst_indicator(r: EpisodeReplay) -> int | None:
    """1 if the episode ended with a false-success termination, 0 if not, None if undecidable (ambiguous)."""
    if r.outcome == "ambiguous_trigger":
        return None
    return 1 if r.outcome in ("early_trigger", "false_trigger") else 0
