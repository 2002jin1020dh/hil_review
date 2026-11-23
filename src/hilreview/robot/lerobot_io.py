"""LeRobot (>= 0.6.1) integration. Runs ONLY on the GPU laptop with `pip install "lerobot[hilserl]>=0.6.1"`.

STATUS: the pure helpers at the top are unit-tested in the sandbox. Every function that imports lerobot is
UNVERIFIED: it was written against the LeRobot 0.6.1 source (read, not executed: no torch in the sandbox).
Before trusting them, run each on 2 episodes and inspect the outputs (see docs/LAPTOP_CHECKS.md, step 5).

API facts relied upon (from the 0.6.1 source):
* reward classifier label key: "next.reward" (lerobot.utils.constants.REWARD), float32 shape (1,)
* runtime: RewardClassifierProcessorStep -> Classifier.predict_reward(images, threshold): probs > threshold,
  applied to observation images WITHOUT the training normalizer (lerobot-train normalizes VISUAL inputs
  MEAN_STD via make_classifier_processor). Hence score_keyframes() offers input_mode "runtime" and "train";
  verify on the laptop whether they differ (open issue in PROJECT_STATUS.md).
* episode boundaries: dataset.meta.episodes[i]["dataset_from_index" / "dataset_to_index"]
* dataset editing: LeRobotDataset.create(...), add_frame(frame), save_episode(), finalize()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from hilreview.data.io import read_jsonl, write_jsonl
from hilreview.data.schema import DataKind, Episode, Keyframe, Label, make_sample_id
from hilreview.data.synthetic import keyframe_indices

DEFAULT_FEATURE_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
REWARD_KEY = "next.reward"
FRAME_TARGET_MODES = ("keyframes_only", "hold_previous", "nearest")

# ------------------------------------------------------------------------------------------------
# Pure helpers (tested in the sandbox)
# ------------------------------------------------------------------------------------------------


def lerobot_source(repo_id: str, episode_index: int) -> str:
    return f"lerobot:{repo_id}#episode={episode_index}"


def parse_lerobot_source(source: str) -> tuple[str, int]:
    m = re.fullmatch(r"lerobot:(.+)#episode=(\d+)", source)
    if not m:
        raise ValueError(f"not a lerobot source string: {source!r}")
    return m.group(1), int(m.group(2))


def frame_targets_from_keyframes(
    n_frames: int, kf_frame_indices: Sequence[int], kf_targets: Sequence[int | None], mode: str = "keyframes_only"
) -> list[int | None]:
    """Expand keyframe training targets (1/0/None) to per-frame targets.

    keyframes_only: only keyframes get a target (default; no assumption about frames between keyframes)
    hold_previous:  each frame takes the target of the most recent keyframe at or before it
    nearest:        each frame takes the target of the nearest keyframe (ties -> earlier keyframe)
    None means "do not use this frame for classifier training".
    """
    if mode not in FRAME_TARGET_MODES:
        raise ValueError(f"mode must be one of {FRAME_TARGET_MODES}")
    if len(kf_frame_indices) != len(kf_targets):
        raise ValueError("keyframe indices and targets differ in length")
    if list(kf_frame_indices) != sorted(set(kf_frame_indices)):
        raise ValueError("keyframe indices must be strictly increasing")
    if kf_frame_indices and not (0 <= kf_frame_indices[0] and kf_frame_indices[-1] < n_frames):
        raise ValueError("keyframe index out of range")
    out: list[int | None] = [None] * n_frames
    if mode == "keyframes_only" or not kf_frame_indices:
        for i, t in zip(kf_frame_indices, kf_targets):
            out[i] = t
        return out
    k = 0
    for f in range(n_frames):
        while k + 1 < len(kf_frame_indices) and kf_frame_indices[k + 1] <= f:
            k += 1
        if mode == "hold_previous":
            out[f] = kf_targets[k] if kf_frame_indices[k] <= f else None
        else:  # nearest
            cand = [k]
            if k + 1 < len(kf_frame_indices):
                cand.append(k + 1)
            best = min(cand, key=lambda j: (abs(kf_frame_indices[j] - f), kf_frame_indices[j]))
            out[f] = kf_targets[best]
    return out


def write_condition_label_file(path: str | Path, keyframes: Iterable[Keyframe], labels: Mapping[str, Label | None],
                               sources: Mapping[str, str], condition: str, replicate: int) -> int:
    """One JSONL row per pool keyframe: what `build_reward_dataset` needs to write next.reward."""
    rows = []
    for kf in keyframes:
        lab = labels.get(kf.sample_id)
        rows.append({
            "condition": condition, "replicate": replicate, "sample_id": kf.sample_id, "episode_id": kf.episode_id,
            "frame_index": kf.frame_index, "label": None if lab is None else lab.value,
            "target": 1 if lab is Label.SUCCESS else (0 if lab is Label.FAILURE else None),
            "source": sources.get(kf.sample_id, "none"),
        })
    return write_jsonl(path, rows)


# ------------------------------------------------------------------------------------------------
# LeRobot I/O (UNVERIFIED: requires lerobot>=0.6.1, torch; run on the laptop)
# ------------------------------------------------------------------------------------------------


def _require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as exc:  # pragma: no cover - sandbox has no lerobot
        raise RuntimeError("lerobot>=0.6.1 is required (GPU laptop only): pip install 'lerobot[hilserl]>=0.6.1'") from exc


def _episode_bounds(ds) -> dict[int, tuple[int, int]]:  # pragma: no cover - needs lerobot
    out = {}
    for i in range(len(ds.meta.episodes)):
        row = ds.meta.episodes[i]
        out[int(row["episode_index"])] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
    return out


def _to_uint8_hwc(img):  # pragma: no cover - needs torch
    import numpy as np

    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr


def _as_feature_array(value, spec):
    """Dataset items come back with shape () for (1,)-features; add_frame validates the declared shape."""
    import numpy as np

    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    dtype = {"float32": np.float32, "float64": np.float64, "int64": np.int64, "int32": np.int32,
             "bool": bool}.get(spec.get("dtype"), arr.dtype)
    return arr.astype(dtype).reshape(tuple(spec["shape"]))


def export_keyframes(  # pragma: no cover - UNVERIFIED
    repo_id: str, root: str | None, out_dir: str | Path, stride: int, session_of: Mapping[int, str],
    mode_of: Mapping[int, str], camera_keys: Sequence[str] | None = None,
    ineligible: Mapping[int, str] | None = None,
) -> tuple[Path, Path]:
    """Write keyframe PNGs + episodes.jsonl + keyframes.jsonl for VLM labeling and review.

    ``session_of``/``mode_of`` map LeRobot episode_index -> session_id / collection_mode (from your
    collection log); every episode must be present in both. ``ineligible`` maps episode_index -> reason.
    """
    _require_lerobot()
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    cams = list(camera_keys or ds.meta.camera_keys)
    out = Path(out_dir)
    episodes, keyframes = [], []
    for ep_idx, (lo, hi) in sorted(_episode_bounds(ds).items()):
        n = hi - lo
        eid = f"ep{ep_idx:04d}"
        reason = (ineligible or {}).get(ep_idx)
        episodes.append(Episode(eid, session_of[ep_idx], mode_of[ep_idx], n, float(ds.meta.fps), DataKind.REAL,
                                lerobot_source(repo_id, ep_idx), reason is None, reason))
        for f in keyframe_indices(n, stride):
            item = ds[lo + f]
            refs = {}
            for cam in cams:
                p = out / "frames" / eid / f"{cam.replace('.', '_')}_{f:06d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_to_uint8_hwc(item[cam])).save(p)
                refs[cam] = str(p)
            keyframes.append(Keyframe(make_sample_id(eid, f), eid, f, f / float(ds.meta.fps), refs))
    write_jsonl(out / "episodes.jsonl", episodes)
    write_jsonl(out / "keyframes.jsonl", keyframes)
    return out / "episodes.jsonl", out / "keyframes.jsonl"


def build_reward_dataset(  # pragma: no cover - UNVERIFIED
    src_repo_id: str, src_root: str | None, episodes_jsonl: str | Path, labels_jsonl: str | Path,
    out_repo_id: str, out_root: str | Path, mode: str = "keyframes_only",
) -> int:
    """Create a NEW LeRobot dataset whose next.reward comes from one condition's labels. Returns #frames written.

    Frames whose target is None (ambiguous / invalid / no label / non-keyframe in keyframes_only mode) are left
    out, so the classifier never trains on them. The source dataset is not modified.
    """
    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(src_repo_id, root=src_root)
    eps = {e.episode_id: e for e in read_jsonl(episodes_jsonl, Episode.from_dict)}
    by_ep: dict[str, list[dict]] = {}
    for r in read_jsonl(labels_jsonl):
        by_ep.setdefault(r["episode_id"], []).append(r)
    features = {k: v for k, v in src.meta.features.items() if k not in DEFAULT_FEATURE_KEYS}
    features[REWARD_KEY] = {"dtype": "float32", "shape": (1,), "names": None}
    dst = LeRobotDataset.create(out_repo_id, fps=src.meta.fps, features=features, root=out_root,
                                robot_type=src.meta.robot_type, use_videos=True)
    bounds = _episode_bounds(src)
    written = 0
    import numpy as np

    for eid in sorted(by_ep):
        rows = sorted(by_ep[eid], key=lambda r: r["frame_index"])
        _, ep_idx = parse_lerobot_source(eps[eid].source)
        lo, hi = bounds[ep_idx]
        targets = frame_targets_from_keyframes(hi - lo, [r["frame_index"] for r in rows],
                                               [r["target"] for r in rows], mode)
        n_ep = 0
        for f, t in enumerate(targets):
            if t is None:
                continue
            item = src[lo + f]
            frame = {}
            for k, spec in features.items():
                if k == REWARD_KEY:
                    frame[k] = np.array([float(t)], dtype=np.float32)
                elif spec.get("dtype") in ("image", "video"):
                    frame[k] = _to_uint8_hwc(item[k])
                elif k in item:
                    frame[k] = _as_feature_array(item[k], spec)
            frame["task"] = item.get("task", "")
            dst.add_frame(frame)
            n_ep += 1
        if n_ep:
            dst.save_episode()
            written += n_ep
    dst.finalize()
    return written


def score_keyframes(  # pragma: no cover - UNVERIFIED
    classifier_path: str, repo_id: str, root: str | None, episodes_jsonl: str | Path, keyframes_jsonl: str | Path,
    episode_ids: Sequence[str], out_jsonl: str | Path, condition: str, replicate: int,
    input_mode: str = "runtime", device: str = "cuda",
) -> int:
    """Score keyframes with a trained LeRobot reward classifier -> rows {condition, replicate, sample_id, p_success}.

    input_mode="runtime" mirrors RewardClassifierProcessorStep (raw dataset images, no normalizer);
    input_mode="train" applies the training preprocessor (MEAN_STD with dataset stats).
    Score the SAME (cropped/resized) dataset the classifier was trained on.
    """
    _require_lerobot()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.rewards import make_reward_pre_post_processors
    from lerobot.rewards.classifier.modeling_classifier import Classifier

    if input_mode not in ("runtime", "train"):
        raise ValueError("input_mode must be 'runtime' or 'train'")
    clf = Classifier.from_pretrained(classifier_path)
    clf.to(device)
    clf.eval()
    image_keys = [k for k in clf.config.input_features if k.startswith("observation.image")]
    ds = LeRobotDataset(repo_id, root=root)
    pre = None
    if input_mode == "train":
        pre, _ = make_reward_pre_post_processors(clf.config, dataset_stats=ds.meta.stats)
    eps = {e.episode_id: e for e in read_jsonl(episodes_jsonl, Episode.from_dict)}
    bounds = _episode_bounds(ds)
    wanted = set(episode_ids)
    rows = []
    with torch.inference_mode():
        for kf in read_jsonl(keyframes_jsonl, Keyframe.from_dict):
            if kf.episode_id not in wanted or not kf.auto_valid:
                continue
            _, ep_idx = parse_lerobot_source(eps[kf.episode_id].source)
            item = ds[bounds[ep_idx][0] + kf.frame_index]
            batch = {k: item[k].unsqueeze(0) for k in image_keys}
            if pre is not None:
                batch = pre(batch)
            images = [batch[k].to(device) for k in image_keys]
            p = float(clf.predict(images).probabilities.reshape(-1)[0])
            rows.append({"condition": condition, "replicate": replicate, "sample_id": kf.sample_id,
                         "p_success": p, "input_mode": input_mode})
    return write_jsonl(out_jsonl, rows)
