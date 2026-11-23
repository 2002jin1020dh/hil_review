import numpy as np
import pytest

from hilreview.data.io import read_jsonl
from hilreview.data.schema import Keyframe, Label, make_sample_id
from hilreview.evaluation.online import summarize_run
from hilreview.robot.adapter import (
    APPROVAL_ENV,
    HardwareNotApproved,
    MockRobotEnv,
    require_hardware_approval,
    rollout_mock_episode,
)
from hilreview.robot.lerobot_io import (
    frame_targets_from_keyframes,
    lerobot_source,
    parse_lerobot_source,
    write_condition_label_file,
)


def test_frame_target_modes():
    idx, tgt = [0, 5, 10, 14], [0, 0, 1, None]
    ko = frame_targets_from_keyframes(15, idx, tgt, "keyframes_only")
    assert ko[0] == 0 and ko[5] == 0 and ko[10] == 1 and ko[14] is None and ko[1] is None
    hp = frame_targets_from_keyframes(15, idx, tgt, "hold_previous")
    assert hp[:10] == [0] * 10 and hp[10:14] == [1] * 4 and hp[14] is None
    nn = frame_targets_from_keyframes(15, idx, tgt, "nearest")
    assert nn[7] == 0 and nn[8] == 1  # 7 is nearer to 5, 8 nearer to 10
    assert nn[12] == 1 and nn[13] is None  # 12: |12-10|=2 < |12-14|=2? tie -> earlier keyframe (10)
    hp2 = frame_targets_from_keyframes(10, [3, 6], [1, 0], "hold_previous")
    assert hp2[:3] == [None] * 3  # frames before the first keyframe get no target
    with pytest.raises(ValueError):
        frame_targets_from_keyframes(10, [5, 3], [0, 1])
    with pytest.raises(ValueError):
        frame_targets_from_keyframes(10, [0, 12], [0, 1])
    with pytest.raises(ValueError):
        frame_targets_from_keyframes(10, [0], [0], "interpolate")


def test_source_roundtrip():
    s = lerobot_source("user/so101_pick_place", 17)
    assert parse_lerobot_source(s) == ("user/so101_pick_place", 17)
    with pytest.raises(ValueError):
        parse_lerobot_source("synthetic:abc#syn0001")


def test_condition_label_file(tmp_path):
    kfs = [Keyframe(make_sample_id("e1", i), "e1", i, i / 10) for i in (0, 5, 10)]
    labels = {kfs[0].sample_id: Label.FAILURE, kfs[1].sample_id: Label.AMBIGUOUS, kfs[2].sample_id: Label.SUCCESS}
    n = write_condition_label_file(tmp_path / "l.jsonl", kfs, labels, {k.sample_id: "vlm" for k in kfs}, "C", 0)
    rows = read_jsonl(tmp_path / "l.jsonl")
    assert n == 3 and [r["target"] for r in rows] == [0, None, 1]


def test_hardware_guard(monkeypatch):
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    with pytest.raises(HardwareNotApproved):
        require_hardware_approval()
    monkeypatch.setenv(APPROVAL_ENV, "yes")
    require_hardware_approval()


def test_mock_env_rejects_unsafe_actions_and_logs_episodes():
    env = MockRobotEnv(seed=1)
    env.reset()
    for bad in (np.array([2.0, 0, 0, 0]), np.array([np.nan, 0, 0, 0]), np.zeros(3)):
        with pytest.raises(ValueError):
            env.step(bad)
    recs = [rollout_mock_episode(MockRobotEnv(seed=s, classifier_fpr=0.2), "mock-run", "eval", s) for s in range(20)]
    summ = summarize_run(recs)
    assert summ["eval_success"]["n"] == 20
    # with a 20% per-step false-positive classifier, some terminations must be false successes
    assert summ["eval_false_success_terminations"]["count"] > 0
    env.close()
    with pytest.raises(RuntimeError):
        env.step(np.zeros(4))


def test_select_label_rows_train_only_and_never_audit():
    import pytest as _pt

    from hilreview.robot.lerobot_io import select_label_rows

    rows = [{"episode_id": e, "frame_index": 0, "target": 1} for e in ("t1", "v1", "a1", "t1")]
    split = {"t1": "train", "v1": "val", "a1": "audit"}
    got = select_label_rows(rows, split)
    assert list(got) == ["t1"] and len(got["t1"]) == 2
    assert sorted(select_label_rows(rows, split, ("train", "val"))) == ["t1", "v1"]
    with _pt.raises(ValueError, match="audit"):
        select_label_rows(rows, split, ("train", "audit"))
    with _pt.raises(ValueError, match="not in the split"):
        select_label_rows([{"episode_id": "x1"}], split)
