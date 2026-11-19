"""SYNTHETIC episode generator for CPU-only pipeline validation (DECISIONS.md D-14).

Nothing produced here is evidence about real robots, VLMs or humans. The generator exists so the
pipeline (splits -> VLM suggestions -> review -> labels -> classifier -> audit metrics) can be tested
end-to-end with known ground truth.

Episode types (``collection_mode``):
  success_attempt: object is placed in the container at an onset time and stays (occasionally it is
                   knocked out again: "revert"). Just before the onset the object hovers over the container.
  near_miss:       object hovers over the container, then ends up outside. Never a success.
                   Hover frames are "fake success" bait for the mock VLM.
  clear_failure:   object never reaches the container.
Keyframe features (6-D) depend on the true state, the hover flag, task progress, a per-session
nuisance offset (e.g. lighting) and noise. They stand in for image embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from hilreview.data.schema import (
    STATE_FAILURE,
    STATE_SUCCESS,
    DataKind,
    Episode,
    Keyframe,
    SyntheticKeyframeTruth,
    make_sample_id,
)
from hilreview.utils.provenance import config_hash
from hilreview.utils.rng import make_rng

N_FEATURES = 6


def keyframe_indices(n_frames: int, stride: int) -> list[int]:
    """Every ``stride``-th frame starting at 0, plus the last frame. Shared with the LeRobot exporter."""
    if n_frames <= 0 or stride <= 0:
        raise ValueError("n_frames and stride must be positive")
    idx = list(range(0, n_frames, stride))
    if idx[-1] != n_frames - 1:
        idx.append(n_frames - 1)
    return idx


@dataclass(frozen=True)
class SyntheticConfig:
    n_episodes: int = 120
    n_sessions: int = 6
    fps: float = 10.0
    duration_s: tuple[float, float] = (12.0, 20.0)
    keyframe_stride: int = 5
    mode_probs: Mapping[str, float] = field(
        default_factory=lambda: {"success_attempt": 0.6, "near_miss": 0.2, "clear_failure": 0.2}
    )
    onset_frac: tuple[float, float] = (0.45, 0.85)
    hover_s: float = 1.5
    revert_prob: float = 0.05
    occluded_prob: float = 0.04
    feature_noise: float = 0.7
    session_offset_sd: float = 0.5
    ineligible_prob: float = 0.03
    auto_invalid_keyframe_prob: float = 0.005
    seed: int = 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["mode_probs"] = dict(self.mode_probs)
        d["duration_s"] = list(self.duration_s)
        d["onset_frac"] = list(self.onset_frac)
        return d


@dataclass
class SyntheticDataset:
    episodes: list[Episode]
    keyframes_by_episode: dict[str, list[Keyframe]]
    truth: dict[str, SyntheticKeyframeTruth]
    config: SyntheticConfig


def _state_fn(mode: str, duration: float, rng: np.random.Generator, cfg: SyntheticConfig):
    """Return (success_intervals, hover_interval) in seconds."""
    if mode == "success_attempt":
        t_on = rng.uniform(*cfg.onset_frac) * duration
        end = duration
        if rng.random() < cfg.revert_prob:
            t_rev = t_on + rng.uniform(1.0, 3.0)
            if t_rev < duration - 0.5:
                end = t_rev
        return [(t_on, end)], (max(0.0, t_on - cfg.hover_s), t_on)
    if mode == "near_miss":
        t_h = rng.uniform(0.45, 0.8) * duration
        return [], (t_h, min(duration, t_h + cfg.hover_s))
    if mode == "clear_failure":
        return [], None
    raise ValueError(f"unknown collection_mode {mode!r}")


def generate_synthetic(cfg: SyntheticConfig) -> SyntheticDataset:
    modes = sorted(cfg.mode_probs)
    probs = np.array([cfg.mode_probs[m] for m in modes], dtype=float)
    if np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("mode_probs must be non-negative with positive sum")
    probs = probs / probs.sum()
    src = f"synthetic:{config_hash(cfg.to_dict())}"
    session_rng = make_rng(cfg.seed, "sessions")
    offsets = session_rng.normal(0.0, cfg.session_offset_sd, size=(cfg.n_sessions, N_FEATURES))

    episodes: list[Episode] = []
    kfs_by_ep: dict[str, list[Keyframe]] = {}
    truth: dict[str, SyntheticKeyframeTruth] = {}
    for i in range(cfg.n_episodes):
        rng = make_rng(cfg.seed, "episode", i)
        eid = f"syn{i:04d}"
        sess = min(cfg.n_sessions - 1, i * cfg.n_sessions // cfg.n_episodes)  # collection order -> sessions
        mode = modes[int(rng.choice(len(modes), p=probs))]
        duration = float(rng.uniform(*cfg.duration_s))
        n_frames = max(2, int(round(duration * cfg.fps)))
        duration = n_frames / cfg.fps
        intervals, hover = _state_fn(mode, duration, rng, cfg)
        eligible = bool(rng.random() >= cfg.ineligible_prob)
        episodes.append(
            Episode(eid, f"session{sess}", mode, n_frames, cfg.fps, DataKind.SYNTHETIC, f"{src}#{eid}", eligible,
                    None if eligible else "synthetic: simulated camera dropout")
        )
        idx = keyframe_indices(n_frames, cfg.keyframe_stride)
        times = [j / cfg.fps for j in idx]
        states = [STATE_SUCCESS if any(a <= t < b for a, b in intervals) else STATE_FAILURE for t in times]
        kf_list = []
        for pos, (j, t, s) in enumerate(zip(idx, times, states)):
            hov = bool(hover is not None and hover[0] <= t < hover[1] and s == STATE_FAILURE)
            prev_s = states[pos - 1] if pos > 0 else s
            next_s = states[pos + 1] if pos + 1 < len(states) else s
            boundary = prev_s != s or next_s != s
            occ_p = cfg.occluded_prob * (3.0 if (hov or boundary) else 1.0)
            occluded = bool(rng.random() < min(1.0, occ_p))
            sv = 1.0 if s == STATE_SUCCESS else 0.0
            hv = 1.0 if hov else 0.0
            prog = t / duration
            base = np.array([1.5 * sv + 0.8 * hv, -1.0 * sv + 0.5 * prog, 0.9 * sv - 0.6 * hv, prog, 0.0, 0.0])
            noise_scale = np.full(N_FEATURES, cfg.feature_noise)
            if occluded:
                noise_scale[:3] *= 2.0
            feats = base + offsets[sess] + rng.normal(0.0, 1.0, N_FEATURES) * noise_scale
            sid = make_sample_id(eid, j)
            kf_list.append(Keyframe(sid, eid, j, t, {"front": f"synthetic://{eid}/{j}"}, tuple(feats.tolist()),
                                    bool(rng.random() >= cfg.auto_invalid_keyframe_prob)))
            truth[sid] = SyntheticKeyframeTruth(sid, s, boundary, hov, occluded)
        kfs_by_ep[eid] = kf_list
    return SyntheticDataset(episodes, kfs_by_ep, truth, cfg)
