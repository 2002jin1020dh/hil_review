"""D-38 HIL-SERL config generation: shared bases, allowed differences, blinding, required fields."""

import json
from pathlib import Path

import pytest

from hilreview.robot.hilserl_config import (
    assert_only_allowed_differences,
    blinding_leaks,
    get_path,
    load_json,
    make_record_config,
    make_train_config,
    missing_fields,
    validate_with_lerobot,
)

ROOT = Path(__file__).resolve().parents[1] / "configs" / "hilserl"


@pytest.fixture
def bases():
    return load_json(ROOT / "env_base.json"), load_json(ROOT / "train_base.json")


@pytest.fixture
def artifacts(tmp_path):
    clf = tmp_path / "store" / "clf_C_rep0" / "pretrained_model"
    demos = tmp_path / "store" / "demos_C_rep0"
    clf.mkdir(parents=True)
    demos.mkdir(parents=True)
    return clf, demos


def test_base_matches_protocol_and_hardware(bases):
    env, train = bases
    assert env["fps"] == 10  # PROTOCOL: keyframes every 5 frames at 10 fps
    assert env["robot"]["port"] == "/dev/so101_follower" and env["robot"]["id"] == "hil_follower"
    assert env["robot"]["cameras"]["front"]["index_or_path"] == "/dev/cam_front"
    assert env["processor"]["inverse_kinematics"]["target_frame_name"] == "gripper_frame_link"
    assert train["wandb"]["enable"] and train["wandb"]["mode"] == "offline"  # D-36
    assert train["policy"]["storage_device"] == "cpu"  # 8 GB GPU cannot hold the buffers
    assert set(train["policy"]["input_features"]) >= {"observation.images.front", "observation.images.wrist"}


def test_record_config_has_no_classifier(bases):
    env, _ = bases
    env = json.loads(json.dumps(env))
    env["processor"]["reward_classifier"]["pretrained_path"] = "/somewhere"
    rec = make_record_config(env, "hil/demos", "/tmp/x", 20)
    assert rec["mode"] == "record" and "type" not in rec["env"]
    assert get_path(rec, "env.processor.reward_classifier.pretrained_path") is None


def test_train_configs_blinded_and_differ_only_where_allowed(bases, artifacts, tmp_path):
    env, train = bases
    clf, demos = artifacts
    a = make_train_config(env, train, run_id="run-1a2b3c4d", seed=1000, runs_root=tmp_path / "runs",
                          classifier_dir=clf, demos_root=demos)
    b = make_train_config(env, train, run_id="run-99aa77bb", seed=1001, runs_root=tmp_path / "runs",
                          classifier_dir=clf, demos_root=demos)
    assert_only_allowed_differences([a, b])
    assert blinding_leaks(a) == []  # real paths contain "C_rep0" but only the run-id symlinks are printed
    lnk = Path(get_path(a, "env.processor.reward_classifier.pretrained_path"))
    assert lnk.is_symlink() and lnk.resolve() == clf.resolve()
    assert Path(a["dataset"]["root"]).resolve() == demos.resolve()
    b["policy"]["online_buffer_capacity"] = 1
    with pytest.raises(ValueError, match="outside the allowed"):
        assert_only_allowed_differences([a, b])


def test_condition_named_run_id_and_leaks_rejected(bases, artifacts, tmp_path):
    env, train = bases
    clf, demos = artifacts
    with pytest.raises(ValueError, match="names a condition"):
        make_train_config(env, train, run_id="C_vlm_only_rep0", seed=1, runs_root=tmp_path, classifier_dir=clf,
                          demos_root=demos)
    cfg = make_train_config(env, train, run_id="run-1a2b3c4d", seed=1, runs_root=tmp_path / "r", classifier_dir=clf,
                            demos_root=demos, link=False)
    cfg["env"]["processor"]["reward_classifier"]["pretrained_path"] = str(clf)  # real path with condition name
    assert blinding_leaks(cfg)


def test_missing_fields(bases, tmp_path):
    env, train = bases
    rec = make_record_config(env, "hil/demos", "/tmp/x", 20)
    assert set(missing_fields(rec, online=False)) == {
        "env.processor.image_preprocessing.crop_params_dict",
        "env.processor.inverse_kinematics.end_effector_bounds",
        "env.processor.reset.fixed_reset_joint_positions"}
    t = make_train_config(env, train, run_id="run-1", seed=1, runs_root=tmp_path, classifier_dir=None,
                          demos_root=None, link=False)
    assert "env.processor.reward_classifier.pretrained_path" in missing_fields(t, online=True)


def test_lerobot_parses_generated_configs(bases, artifacts, tmp_path):
    pytest.importorskip("lerobot")
    env, train = bases
    clf, demos = artifacts
    rec = tmp_path / "record.json"
    rec.write_text(json.dumps(make_record_config(env, "hil/demos", str(tmp_path / "rec"), 20)))
    validate_with_lerobot(rec, "record")
    tr = tmp_path / "train.json"
    tr.write_text(json.dumps(make_train_config(env, train, run_id="run-1a2b3c4d", seed=1000,
                                               runs_root=tmp_path / "runs", classifier_dir=clf, demos_root=demos)))
    cfg = validate_with_lerobot(tr, "train")
    cfg.validate()
    assert cfg.wandb.mode == "offline" and cfg.policy.storage_device == "cpu"
