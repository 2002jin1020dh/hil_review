"""Robot/simulator adapter interface, a SAFE mock, and a hardware approval guard.

No code in this repository commands a physical robot. Real-robot HIL-SERL runs use LeRobot's own
`python -m lerobot.rl.learner` / `python -m lerobot.rl.actor`, started BY THE USER. This module only
defines the minimal interface our tooling expects and a mock used to test online-metric plumbing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from hilreview.evaluation.online import OnlineEpisodeRecord
from hilreview.data.schema import DataKind

APPROVAL_ENV = "HILREVIEW_I_APPROVE_ROBOT_MOTION"


class HardwareNotApproved(RuntimeError):
    pass


def require_hardware_approval() -> None:
    """Gate for any future code path that could move hardware. The human must set the env var explicitly."""
    if os.environ.get(APPROVAL_ENV) != "yes":
        raise HardwareNotApproved(
            f"Refusing to touch hardware: set {APPROVAL_ENV}=yes yourself after checking the workspace is clear, "
            "the emergency stop is reachable and joint/EE bounds are configured."
        )


class RobotEnv(Protocol):
    def reset(self) -> tuple[Any, dict]: ...

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]: ...

    def close(self) -> None: ...


@dataclass
class MockRobotEnv:
    """Pure-software stand-in (no hardware). Episodes succeed after a random number of steps; a fake
    'classifier' fires with a configurable per-step false-positive probability (SYNTHETIC)."""

    max_steps: int = 50
    p_success_step: float = 0.05
    classifier_fpr: float = 0.01
    action_dim: int = 4
    action_bound: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.success = False
        self.closed = False

    def reset(self) -> tuple[np.ndarray, dict]:
        self.t, self.success = 0, False
        return np.zeros(3), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.closed:
            raise RuntimeError("env closed")
        a = np.asarray(action, dtype=float)
        if a.shape != (self.action_dim,) or not np.all(np.isfinite(a)) or np.any(np.abs(a) > self.action_bound):
            raise ValueError(f"unsafe action rejected: {a}")  # mirrors EE-bounds safety in the real pipeline
        self.t += 1
        if not self.success and self.rng.random() < self.p_success_step:
            self.success = True
        fires = self.success or (self.rng.random() < self.classifier_fpr)
        truncated = self.t >= self.max_steps
        info = {"true_success": self.success, "classifier_fired": fires}
        return np.zeros(3), float(fires), bool(fires), truncated and not fires, info

    def close(self) -> None:
        self.closed = True


def rollout_mock_episode(env: MockRobotEnv, run_id: str, phase: str, index: int) -> OnlineEpisodeRecord:
    """Run one mock episode with zero actions and log it like a human observer would (SYNTHETIC)."""
    env.reset()
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(env.action_dim))
        steps += 1
        if terminated or truncated:
            break
    return OnlineEpisodeRecord(run_id, phase, index, "classifier_success" if terminated else "timeout",
                               bool(info["true_success"]), DataKind.SYNTHETIC, total_steps=steps,
                               intervention_steps=0, fps=10.0)
