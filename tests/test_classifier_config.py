"""D-37 reward-classifier train config (no lerobot needed)."""

import json

import pytest

from hilreview.robot.classifier_config import image_features_from_info, make_classifier_train_config


@pytest.fixture
def ds_root(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": {
        "observation.images.front": {"dtype": "video", "shape": [3, 128, 128]},
        "observation.images.wrist": {"dtype": "video", "shape": [3, 128, 128]},
        "observation.state": {"dtype": "float32", "shape": [6]},
        "next.reward": {"dtype": "float32", "shape": [1]}}}))
    return tmp_path


def test_input_features_filled_and_identity_default(ds_root):
    cfg = make_classifier_train_config("x/y", ds_root, "/out", "job")
    rm = cfg["reward_model"]
    assert rm["type"] == "reward_classifier" and rm["num_cameras"] == 2
    assert set(rm["input_features"]) == {"observation.images.front", "observation.images.wrist"}
    assert rm["input_features"]["observation.images.front"] == {"type": "VISUAL", "shape": [3, 128, 128]}
    assert rm["normalization_mapping"]["VISUAL"] == "IDENTITY"  # same input as RewardClassifierProcessorStep
    assert cfg["wandb"] == {"enable": False} and cfg["save_freq"] == cfg["steps"]


def test_camera_subset_and_errors(ds_root):
    assert list(image_features_from_info(ds_root, ["observation.images.front"])) == ["observation.images.front"]
    with pytest.raises(ValueError, match="not in dataset"):
        image_features_from_info(ds_root, ["observation.images.top"])
    with pytest.raises(ValueError):
        make_classifier_train_config("x/y", ds_root, "/o", "j", normalization="MIN_MAX")
