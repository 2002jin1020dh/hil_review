"""lerobot-train config for the reward classifier, identical across conditions (PROTOCOL step 8). No lerobot needed.

Why a generated config file (LeRobot 8c920c42, read in the source):
* lerobot-train does NOT fill ``reward_model.input_features`` from the dataset (make_reward_model passes the
  config as is). Without them the classifier gets no images ("hstack expects a non-empty TensorList").
* Image normalization (D-37): lerobot-train normalizes VISUAL inputs with the classifier preprocessor
  (default MEAN_STD), but the run-time RewardClassifierProcessorStep calls the classifier on raw [0, 1]
  images without that preprocessor. ``normalization="IDENTITY"`` (default here) makes training see exactly
  what the classifier sees at run time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

NORMALIZATIONS = ("IDENTITY", "MEAN_STD")


def image_features_from_info(dataset_root: str | Path, cameras: Sequence[str] | None = None) -> dict[str, list[int]]:
    """{image key: [C, H, W]} from <root>/meta/info.json (keys starting with observation.image)."""
    info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text())
    feats = {k: list(v["shape"]) for k, v in info["features"].items()
             if k.startswith("observation.image") and v.get("dtype") in ("image", "video")}
    if cameras:
        missing = [c for c in cameras if c not in feats]
        if missing:
            raise ValueError(f"cameras not in dataset: {missing}; available {sorted(feats)}")
        feats = {c: feats[c] for c in cameras}
    if not feats:
        raise ValueError(f"no image features in {dataset_root}")
    return feats


def make_classifier_train_config(
    dataset_repo_id: str, dataset_root: str | Path, output_dir: str | Path, job_name: str, *,
    cameras: Sequence[str] | None = None, steps: int = 5000, batch_size: int = 32, seed: int = 1000,
    normalization: str = "IDENTITY", device: str = "cuda", save_freq: int | None = None, num_workers: int = 4,
) -> dict:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    feats = image_features_from_info(dataset_root, cameras)
    return {
        "dataset": {"repo_id": dataset_repo_id, "root": str(dataset_root)},
        "reward_model": {
            "type": "reward_classifier",
            "device": device,
            "num_cameras": len(feats),
            "input_features": {k: {"type": "VISUAL", "shape": s} for k, s in feats.items()},
            "output_features": {"next.reward": {"type": "REWARD", "shape": [1]}},
            "normalization_mapping": {"VISUAL": normalization, "REWARD": "IDENTITY"},
        },
        "output_dir": str(output_dir),
        "job_name": job_name,
        "steps": steps,
        "batch_size": batch_size,
        "save_freq": save_freq or steps,
        "log_freq": 50,
        "seed": seed,
        "num_workers": num_workers,
        "wandb": {"enable": False},
    }
