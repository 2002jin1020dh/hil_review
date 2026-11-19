import pytest

from hilreview.data.schema import DataKind
from hilreview.evaluation.online import OnlineEpisodeRecord, RunRecord, summarize_conditions, summarize_run
from hilreview.experiments.schedule import make_blinded_schedule


def _ep(run, phase, i, end, judged, **kw):
    return OnlineEpisodeRecord(run, phase, i, end, judged, DataKind.SYNTHETIC, **kw)


def test_summarize_run_hand_computed():
    eps = [
        _ep("r1", "train", 0, "classifier_success", False, intervention_steps=20, total_steps=100, fps=10),
        _ep("r1", "train", 1, "classifier_success", True, intervention_steps=0, total_steps=80, fps=10),
        _ep("r1", "train", 2, "timeout", True, intervention_steps=50, total_steps=200, fps=10),  # missed success
        _ep("r1", "train", 3, "classifier_success", None),  # operator could not judge
        _ep("r1", "eval", 0, "classifier_success", True),
        _ep("r1", "eval", 1, "timeout", False),
        _ep("r1", "eval", 2, "classifier_success", False),  # false success during eval
        _ep("r1", "eval", 3, "classifier_success", None),
    ]
    s = summarize_run(eps)
    assert s["eval_success"]["k"] == 1 and s["eval_success"]["n"] == 3 and s["eval_unjudged"] == 1
    t = s["train_false_success_terminations"]
    assert t["count"] == 1 and t["share_of_classifier_terminations"]["n"] == 2
    assert t["unjudged_classifier_terminations"] == 1 and t["per_episode"]["n"] == 3
    assert s["train_missed_success_timeouts"] == 1
    assert s["eval_false_success_terminations"]["count"] == 1
    ti = s["train_interventions"]
    assert ti["episodes_with_intervention"] == 2 and ti["episodes_unknown"] == 1
    assert ti["intervention_steps"] == 70 and ti["intervention_time_s"] == pytest.approx(7.0)
    assert ti["step_intervention_rate"] == pytest.approx(70 / 380)


def test_records_validate():
    with pytest.raises(ValueError):
        _ep("r", "warmup", 0, "timeout", True)
    with pytest.raises(ValueError):
        _ep("r", "train", 0, "crash", True)
    with pytest.raises(ValueError):
        _ep("r", "train", 0, "timeout", True, intervention_steps=5, total_steps=3)
    with pytest.raises(ValueError):
        summarize_run([_ep("a", "eval", 0, "timeout", True), _ep("b", "eval", 0, "timeout", True)])


def test_blinded_schedule_balanced_deterministic_and_opaque():
    conds = ["A_full_review", "B_review25_random", "C_vlm_only"]
    sched, key = make_blinded_schedule(conds, 4, seed=9)
    sched2, key2 = make_blinded_schedule(conds, 4, seed=9)
    assert sched == sched2 and key == key2
    assert len({r["run_id"] for r in sched}) == 12
    assert all(set(r) == {"order", "block", "run_id"} for r in sched)  # no condition leaks into the schedule
    for b in range(4):
        assert sorted(k["condition"] for k in key if k["replicate"] == b) == sorted(conds)
    orders = {tuple(k["condition"] for k in key if k["replicate"] == b) for b in range(4)}
    assert len(orders) > 1  # order is actually randomized across blocks
    with pytest.raises(ValueError):
        make_blinded_schedule(["A", "A"], 1, 0)


def test_summarize_conditions_unblinds_and_handles_failed_runs():
    sched, key = make_blinded_schedule(["A", "C"], 2, seed=1)
    keymap = {k["run_id"]: k for k in key}
    eps, runs = [], []
    for k in key:
        rid = k["run_id"]
        status = "crashed" if (k["condition"] == "C" and k["replicate"] == 1) else "completed"
        runs.append(RunRecord(rid, status, DataKind.SYNTHETIC, wall_clock_s=3600.0))
        n_succ = 8 if k["condition"] == "A" else 5
        for i in range(10):
            eps.append(_ep(rid, "eval", i, "classifier_success" if i < n_succ else "timeout", i < n_succ))
    out = summarize_conditions(eps, runs, keymap)
    assert out["A"]["n_runs_completed"] == 2 and out["A"]["per_seed_eval_success_rate"] == [0.8, 0.8]
    assert out["C"]["n_runs_failed"] == 1 and out["C"]["eval_success_pooled"]["n"] == 10
    with pytest.raises(ValueError):
        summarize_conditions(eps + [_ep("run-unknown", "eval", 0, "timeout", True)], runs, keymap)
