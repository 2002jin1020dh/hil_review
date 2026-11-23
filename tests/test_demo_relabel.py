"""D-29/D-35 demo reward relabel: pure logic (no lerobot needed)."""

import pytest

from hilreview.data.schema import DataKind, Episode
from hilreview.robot.demo_relabel import (
    group_frame_scores,
    human_button_frame,
    relabel_from_scores,
    select_demo_episodes,
)


def test_trigger_uses_next_frame_score():
    # frame 2's image is the first > 0.5 -> it is the observation AFTER transition 1 -> transition 1 ends
    r = relabel_from_scores("e", [0.1, 0.2, 0.9, 0.95, 0.99])
    assert r.trigger_frame == 1
    assert r.n_keep == 2
    assert r.rewards == (0.0, 1.0)
    assert r.dones == (False, True)


def test_frame0_score_is_ignored():
    # the reset observation is never judged at run time
    r = relabel_from_scores("e", [0.99, 0.1, 0.1])
    assert r.trigger_frame is None


def test_frame1_high_ends_first_transition():
    r = relabel_from_scores("e", [0.1, 0.9, 0.1])
    assert (r.trigger_frame, r.n_keep, r.rewards, r.dones) == (0, 1, (1.0,), (True,))


def test_no_trigger_keeps_all_frames_and_never_sets_done():
    # D-35 fix 2: offline buffer has no truncation; done=True would mean "terminal, value 0"
    r = relabel_from_scores("e", [0.1, 0.4, 0.5, 0.3])  # 0.5 is not > 0.5 (strict, like LeRobot)
    assert r.trigger_frame is None
    assert r.n_keep == 4
    assert r.rewards == (0.0,) * 4
    assert r.dones == (False,) * 4


def test_last_frame_score_can_trigger_second_to_last_transition_only():
    r = relabel_from_scores("e", [0.1, 0.1, 0.1, 0.9])
    assert r.trigger_frame == 2 and r.n_keep == 3


def test_single_frame_episode_cannot_trigger():
    r = relabel_from_scores("e", [0.99])
    assert r.trigger_frame is None and r.n_keep == 1 and r.dones == (False,)


def test_empty_episode_rejected():
    with pytest.raises(ValueError):
        relabel_from_scores("e", [])


def test_success_reward_and_threshold_params():
    r = relabel_from_scores("e", [0.0, 0.6, 0.8], threshold=0.7, success_reward=2.0)
    assert r.trigger_frame == 1 and r.rewards == (0.0, 2.0)


def test_lengths_always_match_n_keep():
    for scores in ([0.1] * 7, [0.1, 0.1, 0.9] + [0.1] * 4, [0.9] * 7):
        r = relabel_from_scores("e", scores)
        assert len(r.rewards) == len(r.dones) == r.n_keep <= r.n_src_frames


def test_human_button_frame():
    assert human_button_frame([0, 0, 1, 1], [False, False, False, True]) == 2
    assert human_button_frame([0, 0, 0], [False, False, True]) == 2
    assert human_button_frame([0, 0], [False, False]) is None


def _ep(eid, mode="success_attempt", eligible=True):
    return Episode(eid, "s1", mode, 10, 10.0, DataKind.SYNTHETIC, f"lerobot:x#episode={eid[-1]}", eligible,
                   None if eligible else "bad")


def test_select_demo_episodes_train_success_attempt_only():
    eps = [_ep("e1"), _ep("e2"), _ep("e3", "near_miss"), _ep("e4"), _ep("e5", eligible=False)]
    split = {"e1": "train", "e2": "val", "e3": "train", "e4": "audit", "e5": "train"}
    assert select_demo_episodes(eps, split) == ["e1"]


def test_group_frame_scores_requires_complete_and_unique():
    rows = [{"episode_id": "a", "frame_index": i, "p_success": i / 10, "condition": "A", "replicate": 0}
            for i in range(3)]
    assert group_frame_scores(rows, {"a": 3}) == {"a": [0.0, 0.1, 0.2]}
    with pytest.raises(ValueError, match="missing"):
        group_frame_scores(rows[:2], {"a": 3})
    with pytest.raises(ValueError, match="twice"):
        group_frame_scores(rows + rows[:1], {"a": 3})
    with pytest.raises(ValueError, match="extra"):
        group_frame_scores(rows, {"a": 2})
    with pytest.raises(ValueError, match="mixes"):
        group_frame_scores(rows[:2] + [dict(rows[2], condition="C")], {"a": 3})
