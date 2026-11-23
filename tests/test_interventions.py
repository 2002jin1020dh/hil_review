"""D-26/D-36 intervention extraction: pure logic (no wandb needed)."""

import pytest

from hilreview.data.schema import DataKind
from hilreview.evaluation.interventions import attach_to_records, episodes_from_history
from hilreview.evaluation.online import OnlineEpisodeRecord, summarize_run


def _row(step, rate, flag=None, reward=0.0):
    r = {"train/Interaction step": step, "train/Intervention rate": rate, "train/Episodic reward": reward}
    if flag is not None:
        r["train/Episode intervention"] = flag
    return r


def test_reconstructs_totals_and_steps_and_ignores_other_rows():
    rows = [_row(49, 0.2, 1), {"train/loss_critic": 0.3, "_step": 9}, _row(129, 0.0, 0), _row(199, 0.5, 1, 1.0)]
    eps = episodes_from_history(rows)
    assert [(e.total_steps, e.intervention_steps, e.intervention_flag) for e in eps] == [
        (50, 10, True), (80, 0, False), (70, 35, True)]
    assert [e.seq for e in eps] == [0, 1, 2] and eps[2].episodic_reward == 1.0


def test_rows_sorted_by_step_and_offset():
    eps = episodes_from_history([_row(1099, 0.5), _row(1049, 0.0)], first_step_offset=1000)
    assert [e.total_steps for e in eps] == [50, 50] and eps[1].intervention_steps == 25


def test_inconsistencies_raise():
    with pytest.raises(ValueError, match="integer"):
        episodes_from_history([_row(9, 0.33)])  # 3.3 steps: an episode row is missing or offset wrong
    with pytest.raises(ValueError, match="disagrees"):
        episodes_from_history([_row(9, 0.0, 1)])
    with pytest.raises(ValueError, match="increase"):
        episodes_from_history([_row(9, 0.0), _row(9, 0.0)])


def _rec(run, i, phase="train", human=None):
    return OnlineEpisodeRecord(run, phase, i, "timeout", True, DataKind.SYNTHETIC, human_intervened=human)


def test_attach_fills_only_this_runs_train_episodes_in_order():
    recs = [_rec("r1", 1, human=False), _rec("r1", 0, human=True), _rec("r1", 0, "eval"), _rec("r2", 0)]
    eps = episodes_from_history([_row(49, 0.2, 1), _row(99, 0.1, 1)])
    out, rep = attach_to_records(recs, eps, "r1", fps=10)
    by = {(r.run_id, r.phase, r.episode_index): r for r in out}
    assert (by[("r1", "train", 0)].intervention_steps, by[("r1", "train", 0)].total_steps) == (10, 50)
    assert (by[("r1", "train", 1)].intervention_steps, by[("r1", "train", 1)].total_steps) == (5, 50)
    assert by[("r1", "train", 0)].fps == 10
    assert by[("r1", "eval", 0)].intervention_steps is None and by[("r2", "train", 0)].intervention_steps is None
    assert rep["intervention_time_s"] == pytest.approx(1.5)
    assert rep["human_vs_actor_flag_disagreements"] == [1]  # human said no, actor saw 5 steps
    s = summarize_run([r for r in out if r.run_id == "r1"])
    assert s["train_interventions"]["intervention_time_s"] == pytest.approx(1.5)


def test_attach_refuses_count_mismatch():
    eps = episodes_from_history([_row(49, 0.2, 1)])
    with pytest.raises(ValueError, match="matched by order"):
        attach_to_records([_rec("r1", 0), _rec("r1", 1)], eps, "r1", fps=10)
