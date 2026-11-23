"""HIL-SERL configs for the real SO-101 (LeRobot 8c920c42 gym_manipulator / rl.learner / rl.actor). No lerobot needed
except for ``validate_with_lerobot``.

Design (D-38):
* Two shared base files: configs/hilserl/env_base.json (robot, cameras, gamepad, processors) and
  configs/hilserl/train_base.json (SAC + policy). Every recording and every online run is generated from them.
* Across online runs ONLY these may differ: seed, output_dir, job_name, dataset (the condition's relabeled demos,
  D-29/D-35) and env.processor.reward_classifier.pretrained_path. ``assert_only_allowed_differences`` enforces it.
* Blinding (PROTOCOL 6): LeRobot prints the whole config at start-up, so no condition name may appear in it.
  Paths go through <runs_root>/<run_id>/{classifier,demos} symlinks and names use the opaque run_id only.
* Recording uses the human gamepad buttons only (no classifier): reward_classifier.pretrained_path stays null.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

ALLOWED_RUN_DIFFERENCES = ("seed", "output_dir", "job_name", "dataset", "env.processor.reward_classifier.pretrained_path")
# fields that must be filled before the robot may run (None in the base on purpose)
REQUIRED_BEFORE_ROBOT = {
    "env.processor.image_preprocessing.crop_params_dict": "front/wrist 잘라내기 영역 (물체·용기 확정 후)",
    "env.processor.inverse_kinematics.end_effector_bounds": "끝단 작업 범위 (lerobot-find-joint-limits, gripper_frame_link)",
    "env.processor.reset.fixed_reset_joint_positions": "리셋 자세 (관절 6개, 도 단위)",
}
REQUIRED_FOR_ONLINE = {
    "env.processor.reward_classifier.pretrained_path": "조건의 보상 분류기",
    "dataset.root": "조건의 재라벨된 시연 데이터 (relabel-demos 출력)",
}
CONDITION_WORDS = ("A_", "B_", "C_", "full_review", "review25", "vlm_only", "condition")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def get_path(d: Mapping, dotted: str) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def make_record_config(env_base: Mapping, repo_id: str, root: str, num_episodes: int, device: str = "cuda") -> dict:
    env = copy.deepcopy(dict(env_base))
    env.pop("type", None)  # GymManipulatorConfig.env is HILSerlRobotEnvConfig itself (not a choice type)
    _set_path(env, "processor.reward_classifier.pretrained_path", None)  # demos: human buttons only
    return {"mode": "record", "device": device, "env": env,
            "dataset": {"repo_id": repo_id, "root": str(root), "task": env.get("task") or "",
                        "num_episodes_to_record": int(num_episodes), "push_to_hub": False}}


def make_train_config(env_base: Mapping, train_base: Mapping, *, run_id: str, seed: int, runs_root: str | Path,
                      classifier_dir: str | Path | None, demos_root: str | Path | None,
                      link: bool = True) -> dict:
    """Online-run config for one blinded run. Creates <runs_root>/<run_id>/{classifier,demos} symlinks if ``link``."""
    if any(w.lower() in run_id.lower() for w in CONDITION_WORDS):
        raise ValueError(f"run_id {run_id!r} looks like it names a condition; use the opaque id from the schedule")
    run_dir = Path(runs_root).expanduser() / run_id
    clf_link, demo_link = run_dir / "classifier", run_dir / "demos"
    if link:
        run_dir.mkdir(parents=True, exist_ok=True)
        for lnk, target in ((clf_link, classifier_dir), (demo_link, demos_root)):
            if target is None:
                continue
            target = Path(target).expanduser().resolve()
            if not target.exists():
                raise FileNotFoundError(target)
            if lnk.is_symlink() or lnk.exists():
                if lnk.is_symlink() and Path(os.readlink(lnk)) == target:
                    continue
                raise FileExistsError(f"{lnk} already exists and points elsewhere")
            lnk.symlink_to(target, target_is_directory=True)
    cfg = copy.deepcopy(dict(train_base))
    cfg["env"] = copy.deepcopy(dict(env_base))
    cfg["seed"] = int(seed)
    cfg["job_name"] = run_id
    cfg["output_dir"] = str(run_dir / "output")
    cfg["dataset"] = {"repo_id": f"hil/{run_id}", "root": str(demo_link) if demos_root is not None else None,
                      "use_imagenet_stats": False}
    _set_path(cfg, "env.processor.reward_classifier.pretrained_path",
              str(clf_link) if classifier_dir is not None else None)
    return cfg


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, Mapping):
        out = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: d}


def diff_paths(a: Mapping, b: Mapping) -> set[str]:
    fa, fb = _flatten(a), _flatten(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, object()) != fb.get(k, object())}


def assert_only_allowed_differences(configs: Iterable[Mapping]) -> None:
    configs = list(configs)
    for c in configs[1:]:
        bad = sorted(p for p in diff_paths(configs[0], c)
                     if not any(p == a or p.startswith(a + ".") for a in ALLOWED_RUN_DIFFERENCES))
        if bad:
            raise ValueError(f"online runs differ outside the allowed fields: {bad[:10]}")


def missing_fields(cfg: Mapping, online: bool) -> dict[str, str]:
    """Fields still null that must be set before the robot runs (env is under 'env' in both config kinds)."""
    req = dict(REQUIRED_BEFORE_ROBOT)
    if online:
        req.update(REQUIRED_FOR_ONLINE)
    return {k: why for k, why in req.items() if get_path(cfg, k) in (None, {}, [])}


def blinding_leaks(cfg: Mapping) -> list[str]:
    """Config strings that mention a condition (would be printed on the operator's screen)."""
    leaks = []
    for k, v in _flatten(cfg).items():
        if isinstance(v, str) and any(w.lower() in v.lower() for w in CONDITION_WORDS):
            leaks.append(f"{k}={v}")
    return leaks


def validate_with_lerobot(path: str | Path, kind: str) -> Any:  # pragma: no cover - needs lerobot (tests skip otherwise)
    """Parse the file with LeRobot's own config classes (no robot connection). Returns the parsed config."""
    import draccus

    if kind == "record":
        from lerobot.rl.gym_manipulator import GymManipulatorConfig as cls
    elif kind == "train":
        from lerobot.rl.train_rl import TrainRLServerPipelineConfig as cls
    else:
        raise ValueError("kind must be 'record' or 'train'")
    import lerobot.cameras.opencv  # noqa: F401  register choice types (actor/learner import these too)
    import lerobot.robots.so_follower  # noqa: F401
    import lerobot.teleoperators.gamepad  # noqa: F401

    return draccus.parse(config_class=cls, config_path=str(path), args=[])
