"""SYNTHETIC demo data with rendered toy images, for practising the review workflow before real data exists.

The images are cartoons (a red cube, a blue container, a dark gripper). Labels, VLM suggestions and timings
from a demo session are NOT research data; every record is tagged data_kind="synthetic".
"""

from __future__ import annotations

import json
from pathlib import Path

from hilreview.data.io import write_jsonl
from hilreview.data.schema import Keyframe
from hilreview.data.splits import SplitSpec, assign_splits, check_split_integrity
from hilreview.data.synthetic import SyntheticConfig, generate_synthetic
from hilreview.labeling.vlm import MockVLMLabeler, MockVLMProfile


def _render(state: str, hover: bool, occluded: bool, progress: float, size=(160, 120)):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (150, 150, 150))
    d = ImageDraw.Draw(img)
    d.rectangle([98, 58, 142, 98], outline=(30, 60, 200), width=3)  # container
    if state == "S":
        cx, cy = 120, 80
    elif hover:
        cx, cy = 120, 42
    else:
        cx, cy = 30 + int(60 * min(progress, 1.0)), 85 - int(25 * min(progress, 1.0))
    d.rectangle([cx - 7, cy - 7, cx + 7, cy + 7], fill=(200, 30, 30))  # cube
    gy = cy - 20 if state != "S" else 40
    d.rectangle([cx - 10, gy - 10, cx + 10, gy], fill=(40, 40, 40))  # gripper
    if occluded:
        d.rectangle([cx - 14, cy - 14, cx + 14, cy + 14], fill=(90, 90, 90))
    return img


def make_demo_data(out: Path, n_episodes: int = 12, seed: int = 0) -> dict[str, Path]:
    out = Path(out)
    cfg = SyntheticConfig(n_episodes=n_episodes, n_sessions=2, duration_s=(8.0, 10.0), keyframe_stride=5,
                          ineligible_prob=0.0, auto_invalid_keyframe_prob=0.0, seed=seed)
    ds = generate_synthetic(cfg)
    kfs_out: list[Keyframe] = []
    for e in ds.episodes:
        for kf in ds.keyframes_by_episode[e.episode_id]:
            t = ds.truth[kf.sample_id]
            p = out / "frames" / e.episode_id / f"front_{kf.frame_index:06d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            _render(t.state, t.hover, t.occluded, kf.timestamp_s / e.duration_s).save(p)
            kfs_out.append(Keyframe(kf.sample_id, kf.episode_id, kf.frame_index, kf.timestamp_s, {"front": str(p)},
                                    None, kf.auto_valid))
    vlm = MockVLMLabeler(ds.truth, MockVLMProfile(), seed).suggest(kfs_out)
    spec = SplitSpec(seed=seed)
    split = assign_splits(ds.episodes, spec)
    counts = check_split_integrity(ds.episodes, split)
    paths = {
        "episodes": out / "episodes.jsonl",
        "keyframes": out / "keyframes.jsonl",
        "vlm_suggestions": out / "vlm_suggestions.jsonl",
        "split_file": out / "splits.json",
    }
    write_jsonl(paths["episodes"], ds.episodes)
    write_jsonl(paths["keyframes"], kfs_out)
    write_jsonl(paths["vlm_suggestions"], vlm)
    paths["split_file"].write_text(json.dumps({"split_spec": spec.to_dict(), "counts": counts,
                                               "episode_split": split}, indent=2, sort_keys=True))
    return paths
