"""D-29 (수정 D-35): 시연 데이터(offline buffer)의 보상을 각 조건의 분류기로 다시 매기기.

왜 필요한가: LeRobot 학습기는 배치의 절반(online_ratio=0.5)을 시연 데이터에서 뽑고, 그 보상은 데이터셋에
저장된 값(녹화 때 사람이 누른 성공 버튼)을 그대로 씁니다. 그대로 두면 조건 C에도 사람 라벨이 섞입니다.

규칙 (LeRobot 8c920c42 코드를 읽고, 이 저장소 테스트에서 실제 ReplayBuffer로 확인):
* 녹화(gym_manipulator record 모드)는 프레임 t에 [행동 전 관측 s_t, 행동 a_t, 그 결과의 reward/done]을 저장합니다.
  온라인에서는 RewardClassifierProcessorStep이 행동 뒤의 새 관측 s_{t+1}을 채점합니다.
  -> 프레임 t의 보상·종료는 **프레임 t+1 이미지의 점수**로 정합니다 (D-35 수정 1). 프레임 0(리셋 직후) 점수는
     쓰지 않고, 마지막 프레임 다음 관측은 저장되지 않으므로 마지막 프레임에서는 판정하지 않습니다.
* 처음으로 p > threshold(엄격한 '>', LeRobot predict_reward와 동일)인 전이 t*에서 reward=success_reward,
  done=True. t* 다음 프레임은 버립니다.
* 판정이 없으면 reward는 모두 0, done은 모두 False (D-35 수정 2). 오프라인 버퍼는 truncation을 지원하지 않아
  (buffer.py: truncated = done) done=True는 "진짜 끝(이후 가치 0)"이 됩니다. done=False이면 마지막 프레임의
  next_state가 자기 자신이 되어, 온라인의 시간 초과(truncated, 가치 계속 추정)와 가장 가깝습니다.
* 녹화 때 사람이 누른 성공 버튼 값은 어떤 조건에서도 학습에 쓰지 않습니다. 기록 파일에 비교용으로만 남깁니다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from hilreview.data.io import read_jsonl, write_jsonl
from hilreview.data.schema import Episode
from hilreview.evaluation.termination import first_trigger

DEMO_COLLECTION_MODE = "success_attempt"
DEMO_SPLIT = "train"
REWARD_KEY = "next.reward"
DONE_KEY = "next.done"

# ------------------------------------------------------------------------------------------------
# Pure logic (no lerobot needed)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoRelabel:
    episode_id: str
    n_src_frames: int
    n_keep: int  # frames 0..n_keep-1 are kept
    trigger_frame: int | None  # transition index t* (reward=1, done=True), or None
    rewards: tuple[float, ...]  # length n_keep
    dones: tuple[bool, ...]  # length n_keep

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("rewards")
        d.pop("dones")
        return d


def relabel_from_scores(
    episode_id: str, frame_scores: Sequence[float | None], threshold: float = 0.5, success_reward: float = 1.0
) -> DemoRelabel:
    """Per-frame classifier scores (frame i = image of frame i) -> per-transition reward/done (see module doc)."""
    n = len(frame_scores)
    if n == 0:
        raise ValueError(f"{episode_id}: empty episode")
    # transition t is judged by the score of the observation AFTER the action, i.e. frame t+1
    t_star = first_trigger(list(frame_scores[1:]), threshold, debounce_k=1)
    if t_star is None:
        return DemoRelabel(episode_id, n, n, None, (0.0,) * n, (False,) * n)
    n_keep = t_star + 1
    rewards = (0.0,) * t_star + (float(success_reward),)
    dones = (False,) * t_star + (True,)
    return DemoRelabel(episode_id, n, n_keep, t_star, rewards, dones)


def human_button_frame(rewards: Sequence[float], dones: Sequence[bool]) -> int | None:
    """First frame where the recording itself ended or got reward (human Y button). For comparison ONLY."""
    for i, (r, d) in enumerate(zip(rewards, dones)):
        if r > 0 or d:
            return i
    return None


def select_demo_episodes(
    episodes: Iterable[Episode], split_map: Mapping[str, str], collection_mode: str = DEMO_COLLECTION_MODE
) -> list[str]:
    """D-29: all train-split success_attempt episodes (same list for every condition; never val/audit)."""
    out = []
    for e in episodes:
        if split_map.get(e.episode_id) == DEMO_SPLIT and e.collection_mode == collection_mode and e.eligible:
            out.append(e.episode_id)
    return sorted(out)


def group_frame_scores(rows: Iterable[Mapping], n_frames_of: Mapping[str, int]) -> dict[str, list[float]]:
    """Score rows {episode_id, frame_index, p_success} -> per-episode list. Requires every frame to be scored once."""
    by_ep: dict[str, dict[int, float]] = defaultdict(dict)
    meta: set[tuple] = set()
    for r in rows:
        eid, f = r["episode_id"], int(r["frame_index"])
        if f in by_ep[eid]:
            raise ValueError(f"{eid}: frame {f} scored twice")
        by_ep[eid][f] = float(r["p_success"])
        meta.add((r.get("condition"), r.get("replicate"), r.get("input_mode")))
    if len(meta) > 1:
        raise ValueError(f"score file mixes conditions/replicates/input modes: {sorted(map(str, meta))}")
    out = {}
    for eid, n in n_frames_of.items():
        got = by_ep.get(eid, {})
        missing = [f for f in range(n) if f not in got]
        extra = sorted(set(got) - set(range(n)))
        if missing:
            raise ValueError(f"{eid}: frames 0..{n - 1} need scores; missing {missing[:5]} ({len(missing)} total)")
        if extra:
            raise ValueError(f"{eid}: scores for frames outside 0..{n - 1}: extra {extra[:5]}")
        out[eid] = [got[f] for f in range(n)]
    return out


# ------------------------------------------------------------------------------------------------
# LeRobot I/O (needs lerobot; run in the `lerobot` conda env)
# ------------------------------------------------------------------------------------------------


def _bounds_and_ids(ds, episodes_jsonl, episode_ids):  # pragma: no cover - needs lerobot
    from hilreview.robot.lerobot_io import _episode_bounds, parse_lerobot_source

    eps = {e.episode_id: e for e in read_jsonl(episodes_jsonl, Episode.from_dict)}
    bounds = _episode_bounds(ds)
    out = {}
    for eid in episode_ids:
        _, ep_idx = parse_lerobot_source(eps[eid].source)
        out[eid] = bounds[ep_idx]
    return out


def score_demo_frames(  # pragma: no cover - needs lerobot + a trained classifier
    classifier_path: str, repo_id: str, root: str | None, episodes_jsonl: str | Path, episode_ids: Sequence[str],
    out_jsonl: str | Path, condition: str, replicate: int, input_mode: str = "runtime", device: str = "cuda",
    batch_size: int = 64,
) -> int:
    """Score EVERY frame of the given episodes -> rows {condition, replicate, episode_id, frame_index, p_success}.

    input_mode="runtime" (default, D-29): the same input as RewardClassifierProcessorStep at run time — the
    recorded (already cropped/resized) observation images, no training normalizer.
    """
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
    pre = make_reward_pre_post_processors(clf.config, dataset_stats=ds.meta.stats)[0] if input_mode == "train" else None
    rows = []
    with torch.inference_mode():
        for eid, (lo, hi) in _bounds_and_ids(ds, episodes_jsonl, episode_ids).items():
            for start in range(lo, hi, batch_size):
                idx = list(range(start, min(start + batch_size, hi)))
                items = [ds[i] for i in idx]
                batch = {k: torch.stack([it[k] for it in items]) for k in image_keys}
                if pre is not None:
                    batch = pre(batch)
                probs = clf.predict([batch[k].to(device) for k in image_keys]).probabilities.reshape(-1)
                for i, p in zip(idx, probs.tolist()):
                    rows.append({"condition": condition, "replicate": replicate, "episode_id": eid,
                                 "frame_index": i - lo, "p_success": float(p), "input_mode": input_mode,
                                 "classifier": str(classifier_path)})
    return write_jsonl(out_jsonl, rows)


def write_relabeled_demo_dataset(  # pragma: no cover - exercised by tests/test_demo_relabel_lerobot.py
    src_repo_id: str, src_root: str | None, episodes_jsonl: str | Path, episode_ids: Sequence[str],
    scores_jsonl: str | Path, out_repo_id: str, out_root: str | Path, manifest_path: str | Path,
    threshold: float = 0.5, success_reward: float = 1.0, image_storage: str = "image",
) -> list[dict]:
    """Create a NEW dataset with only the demo episodes, next.reward/next.done rewritten, frames after t* dropped.

    image_storage="image" stores decoded frames losslessly (no second video compression); "video" re-encodes.
    The source dataset is never modified. Returns the manifest rows (also written to ``manifest_path``).
    """
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from hilreview.robot.lerobot_io import DEFAULT_FEATURE_KEYS, _as_feature_array, _to_uint8_hwc

    if image_storage not in ("image", "video"):
        raise ValueError("image_storage must be 'image' or 'video'")
    src = LeRobotDataset(src_repo_id, root=src_root)
    for k in (REWARD_KEY, DONE_KEY, "action"):
        if k not in src.meta.features:
            raise ValueError(f"source dataset has no '{k}' feature (is it a gym_manipulator record dataset?)")
    bounds = _bounds_and_ids(src, episodes_jsonl, episode_ids)
    scores = group_frame_scores(read_jsonl(scores_jsonl), {e: hi - lo for e, (lo, hi) in bounds.items()})

    features = {}
    for k, spec in src.meta.features.items():
        if k in DEFAULT_FEATURE_KEYS:
            continue
        spec = dict(spec)
        if spec.get("dtype") in ("image", "video"):
            spec["dtype"] = image_storage
        features[k] = spec
    dst = LeRobotDataset.create(out_repo_id, fps=src.meta.fps, features=features, root=out_root,
                                robot_type=src.meta.robot_type, use_videos=image_storage == "video")
    manifest = []
    for eid in episode_ids:
        lo, hi = bounds[eid]
        rel = relabel_from_scores(eid, scores[eid], threshold, success_reward)
        src_r, src_d = [], []
        for f in range(rel.n_src_frames):
            it = src[lo + f]
            src_r.append(float(it[REWARD_KEY].reshape(-1)[0]))
            src_d.append(bool(it[DONE_KEY].reshape(-1)[0]))
            if f >= rel.n_keep:
                continue
            frame = {}
            for k, spec in features.items():
                if k == REWARD_KEY:
                    frame[k] = np.array([rel.rewards[f]], dtype=np.float32)
                elif k == DONE_KEY:
                    frame[k] = np.array([rel.dones[f]], dtype=bool)
                elif spec["dtype"] in ("image", "video"):
                    frame[k] = _to_uint8_hwc(it[k])
                else:
                    frame[k] = _as_feature_array(it[k], spec)
            frame["task"] = it["task"]
            dst.add_frame(frame)
        dst.save_episode()
        row = rel.to_dict() | {"human_button_frame": human_button_frame(src_r, src_d), "threshold": threshold}
        manifest.append(row)
    dst.finalize()
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps({"rule": "D-29/D-35", "src_repo_id": src_repo_id,
                                               "scores": str(scores_jsonl), "episodes": manifest},
                                              indent=2, ensure_ascii=False))
    return manifest
