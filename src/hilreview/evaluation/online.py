"""Online (real-robot) episode logs and metrics.

Success is ALWAYS judged by the human observer, never by the reward classifier. The human logs, for every
episode, how it ended (end_reason) and whether the task was truly accomplished (human_judged_success,
None = could not judge). From these:
  * eval success rate (phase="eval"; policy runs without interventions) with Wilson interval,
  * online false-success terminations: end_reason == "classifier_success" but human says failure,
  * missed successes: episode timed out although the human saw success,
  * intervention episodes / steps / seconds (from LeRobot actor per-episode counts: intervention steps / fps),
  * per-run and per-condition (per-seed) summaries with between-seed SD.
Unknown values are excluded from denominators and counted, never imputed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from hilreview.data.schema import DataKind
from hilreview.evaluation.metrics import summarize_values, wilson_interval

END_REASONS = ("classifier_success", "timeout", "human_success_key", "human_failure_key", "abort")
PHASES = ("train", "eval")
RUN_STATUSES = ("completed", "aborted", "crashed")


@dataclass(frozen=True)
class OnlineEpisodeRecord:
    run_id: str
    phase: str
    episode_index: int
    end_reason: str
    human_judged_success: bool | None
    data_kind: DataKind = DataKind.REAL
    duration_s: float | None = None
    intervention_steps: int | None = None
    total_steps: int | None = None
    fps: float | None = None
    started_at: str | None = None
    notes: str = ""
    logged_at: str | None = None  # when the human logged the outcome (episode-outcome logger)
    human_intervened: bool | None = None  # human-reported: was there any intervention in this episode?

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id required")
        if self.human_intervened not in (True, False, None):
            raise ValueError("human_intervened must be True/False/None")
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if self.end_reason not in END_REASONS:
            raise ValueError(f"end_reason must be one of {END_REASONS}")
        if self.human_judged_success not in (True, False, None):
            raise ValueError("human_judged_success must be True/False/None")
        if self.intervention_steps is not None and self.total_steps is not None:
            if not 0 <= self.intervention_steps <= self.total_steps:
                raise ValueError("need 0 <= intervention_steps <= total_steps")
        object.__setattr__(self, "data_kind", DataKind(self.data_kind))

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["data_kind"] = self.data_kind.value
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OnlineEpisodeRecord":
        return cls(**{**d, "data_kind": DataKind(d.get("data_kind", "real"))})


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: str
    data_kind: DataKind = DataKind.REAL
    wall_clock_s: float | None = None
    online_steps: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"status must be one of {RUN_STATUSES}")
        object.__setattr__(self, "data_kind", DataKind(self.data_kind))

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["data_kind"] = self.data_kind.value
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecord":
        return cls(**{**d, "data_kind": DataKind(d.get("data_kind", "real"))})


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": wilson_interval(k, n)}


def summarize_run(episodes: Sequence[OnlineEpisodeRecord]) -> dict:
    out: dict[str, Any] = {}
    run_ids = {e.run_id for e in episodes}
    if len(run_ids) > 1:
        raise ValueError(f"summarize_run got episodes from several runs: {sorted(run_ids)}")
    ev = [e for e in episodes if e.phase == "eval"]
    judged = [e for e in ev if e.human_judged_success is not None]
    out["eval_success"] = _rate(sum(1 for e in judged if e.human_judged_success), len(judged))
    out["eval_unjudged"] = len(ev) - len(judged)
    out["eval_interventions_present"] = sum(1 for e in ev if (e.intervention_steps or 0) > 0)  # should be 0
    for phase in PHASES:
        eps = [e for e in episodes if e.phase == phase]
        clf = [e for e in eps if e.end_reason == "classifier_success"]
        clf_j = [e for e in clf if e.human_judged_success is not None]
        fst = sum(1 for e in clf_j if e.human_judged_success is False)
        timeouts_j = [e for e in eps if e.end_reason == "timeout" and e.human_judged_success is not None]
        out[f"{phase}_n_episodes"] = len(eps)
        out[f"{phase}_false_success_terminations"] = {
            "count": fst,
            "per_episode": _rate(fst, sum(1 for e in eps if e.human_judged_success is not None)),
            "share_of_classifier_terminations": _rate(fst, len(clf_j)),
            "unjudged_classifier_terminations": len(clf) - len(clf_j),
        }
        out[f"{phase}_missed_success_timeouts"] = sum(1 for e in timeouts_j if e.human_judged_success)
    tr = [e for e in episodes if e.phase == "train"]
    known = [e for e in tr if e.intervention_steps is not None]
    with_fps = [e for e in known if e.fps]
    tot_known = [e for e in known if e.total_steps]
    reported = [e for e in tr if e.human_intervened is not None]
    out["train_interventions_human_reported"] = {
        "episodes_with_intervention": sum(1 for e in reported if e.human_intervened),
        "episodes_reported": len(reported),
        "episodes_unreported": len(tr) - len(reported),
    }
    out["train_interventions"] = {
        "episodes_with_intervention": sum(1 for e in known if e.intervention_steps > 0),
        "episodes_unknown": len(tr) - len(known),
        "intervention_steps": sum(e.intervention_steps for e in known),
        "intervention_time_s": (sum(e.intervention_steps / e.fps for e in with_fps) if with_fps else None),
        "intervention_time_unknown_episodes": len(known) - len(with_fps),
        "step_intervention_rate": (sum(e.intervention_steps for e in tot_known) / sum(e.total_steps for e in tot_known)
                                   if tot_known else None),
    }
    return out


def summarize_conditions(
    episodes: Sequence[OnlineEpisodeRecord],
    runs: Sequence[RunRecord],
    key: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Unblind with ``key`` (run_id -> {condition, replicate}) and aggregate per condition."""
    by_run: dict[str, list[OnlineEpisodeRecord]] = defaultdict(list)
    for e in episodes:
        by_run[e.run_id].append(e)
    run_by_id = {r.run_id: r for r in runs}
    unknown = (set(by_run) | set(run_by_id)) - set(key)
    if unknown:
        raise ValueError(f"runs missing from the blinding key: {sorted(unknown)}")
    per_cond: dict[str, dict[str, Any]] = defaultdict(lambda: {"runs": []})
    for run_id, info in sorted(key.items()):
        cond = info["condition"]
        rr = run_by_id.get(run_id)
        s = summarize_run(by_run.get(run_id, []))
        per_cond[cond]["runs"].append({"run_id": run_id, "replicate": info.get("replicate"),
                                       "status": rr.status if rr else "not_run",
                                       "wall_clock_s": rr.wall_clock_s if rr else None, "summary": s})
    out = {}
    for cond, d in per_cond.items():
        runs_c = d["runs"]
        done = [r for r in runs_c if r["status"] == "completed"]
        rates = [r["summary"]["eval_success"]["rate"] for r in done]
        k = sum(r["summary"]["eval_success"]["k"] for r in done)
        n = sum(r["summary"]["eval_success"]["n"] for r in done)
        fst = sum(r["summary"]["train_false_success_terminations"]["count"] for r in done)
        out[cond] = {
            "n_runs_planned": len(runs_c),
            "n_runs_completed": len(done),
            "n_runs_failed": sum(1 for r in runs_c if r["status"] in ("aborted", "crashed")),
            "per_seed_eval_success_rate": rates,
            "eval_success_rate_across_seeds": summarize_values(rates),
            # pooled interval ignores between-seed clustering; report it next to the per-seed values only
            "eval_success_pooled": _rate(k, n),
            "train_false_success_terminations_total": fst,
            "wall_clock_s": summarize_values([r["wall_clock_s"] for r in done]),
            "train_intervention_time_s": summarize_values(
                [r["summary"]["train_interventions"]["intervention_time_s"] for r in done]),
            "runs": runs_c,
        }
    return out
