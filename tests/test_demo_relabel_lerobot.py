"""D-29/D-35 end-to-end with REAL LeRobot (8c920c42): fake record-mode dataset -> relabel -> ReplayBuffer.

Runs only where lerobot is installed (the `lerobot` conda env):
  ~/miniforge3/envs/lerobot/bin/python -m pytest -q tests/test_demo_relabel_lerobot.py
"""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.rl.buffer import ReplayBuffer  # noqa: E402

from hilreview.data.io import write_jsonl  # noqa: E402
from hilreview.data.schema import DataKind, Episode  # noqa: E402
from hilreview.robot.demo_relabel import select_demo_episodes, write_relabeled_demo_dataset  # noqa: E402

H = W = 32
CAM = "observation.images.front"
STATE = "observation.state"
# episode_index -> (n_frames, frame with the human Y button, per-frame classifier scores)
SPEC = {
    0: (6, 4, [0.1, 0.2, 0.3, 0.9, 0.9, 0.9]),  # classifier: frame 3 -> transition 2 ends, keep 3 frames
    1: (5, 4, [0.1, 0.1, 0.4, 0.5, 0.2]),  # classifier never > 0.5 -> keep all, done never set
    2: (4, 3, [0.9, 0.9, 0.9, 0.9]),  # val split -> must not appear in the demo dataset
}


def _pixel(ep, f):
    return 20 + 40 * ep + 10 * f  # distinct, well separated so lossy video still identifies the frame


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """A dataset with the exact feature layout of lerobot.rl.gym_manipulator record mode (video images)."""
    root = tmp_path_factory.mktemp("src") / "ds"
    features = {
        "action": {"dtype": "float32", "shape": (4,), "names": ["delta_x", "delta_y", "delta_z", "gripper"]},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.discrete_penalty": {"dtype": "float32", "shape": (1,), "names": ["discrete_penalty"]},
        STATE: {"dtype": "float32", "shape": (6,), "names": None},
        CAM: {"dtype": "video", "shape": (3, H, W), "names": ["channels", "height", "width"]},
    }
    ds = LeRobotDataset.create("test/src", fps=10, features=features, root=root, use_videos=True)
    for ep, (n, button, _) in SPEC.items():
        for f in range(n):
            img = torch.full((3, H, W), _pixel(ep, f) / 255.0)  # CHW float like the env observation
            ds.add_frame({
                "action": np.array([f, ep, -f, 1.0], dtype=np.float32),
                "next.reward": np.array([1.0 if f >= button else 0.0], dtype=np.float32),
                "next.done": np.array([f == n - 1], dtype=bool),  # recording ends at the last frame
                "complementary_info.discrete_penalty": np.array([0.01 * f], dtype=np.float32),
                STATE: np.full(6, 100 * ep + f, dtype=np.float32),
                CAM: img.numpy(),
                "task": "put the object into the container",
            })
        ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture(scope="module")
def relabeled(src, tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    eps = [Episode(f"ep{i:04d}", "s1", "success_attempt", n, 10.0, DataKind.SYNTHETIC,
                   f"lerobot:test/src#episode={i}") for i, (n, _, _) in SPEC.items()]
    write_jsonl(work / "episodes.jsonl", eps)
    split = {"ep0000": "train", "ep0001": "train", "ep0002": "val"}
    ids = select_demo_episodes(eps, split)
    assert ids == ["ep0000", "ep0001"]
    rows = [{"condition": "C", "replicate": 0, "episode_id": f"ep{i:04d}", "frame_index": f, "p_success": p,
             "input_mode": "runtime"} for i, (_, _, sc) in SPEC.items() for f, p in enumerate(sc)]
    write_jsonl(work / "scores.jsonl", rows)
    out_root = work / "demo_C"
    manifest = write_relabeled_demo_dataset("test/src", str(src), work / "episodes.jsonl", ids, work / "scores.jsonl",
                                            "test/demo_C", out_root, work / "manifest.json")
    return out_root, manifest, work


def _episodes(ds):
    by = {}
    for i in range(len(ds)):
        it = ds[i]
        by.setdefault(int(it["episode_index"]), []).append(it)
    return by


def test_manifest(relabeled):
    _, manifest, work = relabeled
    m = {r["episode_id"]: r for r in manifest}
    assert (m["ep0000"]["trigger_frame"], m["ep0000"]["n_keep"], m["ep0000"]["human_button_frame"]) == (2, 3, 4)
    assert (m["ep0001"]["trigger_frame"], m["ep0001"]["n_keep"], m["ep0001"]["human_button_frame"]) == (None, 5, 4)
    assert json.loads((work / "manifest.json").read_text())["rule"] == "D-29/D-35"


def test_dataset_rewards_dones_and_dropped_frames(relabeled):
    out_root, _, _ = relabeled
    ds = LeRobotDataset("test/demo_C", root=out_root)
    eps = _episodes(ds)
    assert sorted(eps) == [0, 1] and len(ds) == 3 + 5
    r0 = [float(it["next.reward"]) for it in eps[0]]
    d0 = [bool(it["next.done"]) for it in eps[0]]
    assert r0 == [0.0, 0.0, 1.0] and d0 == [False, False, True]
    r1 = [float(it["next.reward"]) for it in eps[1]]
    d1 = [bool(it["next.done"]) for it in eps[1]]
    assert r1 == [0.0] * 5 and d1 == [False] * 5  # human button (frame 4) NOT used; no terminal done


@pytest.mark.parametrize("new_ep,src_ep", [(0, 0), (1, 1)])
def test_other_features_copied_frame_aligned(relabeled, new_ep, src_ep):
    out_root, _, _ = relabeled
    ds = LeRobotDataset("test/demo_C", root=out_root)
    for f, it in enumerate(_episodes(ds)[new_ep]):
        assert it["action"].tolist() == [f, src_ep, -f, 1.0]
        assert float(it["complementary_info.discrete_penalty"]) == pytest.approx(0.01 * f)
        assert it[STATE].tolist() == [100 * src_ep + f] * 6
        img = it[CAM]
        assert tuple(img.shape) == (3, H, W)
        assert abs(float(img.mean()) * 255 - _pixel(src_ep, f)) < 6  # same frame (source was lossy video)


def test_replay_buffer_sees_relabeled_transitions(relabeled):
    out_root, _, _ = relabeled
    ds = LeRobotDataset("test/demo_C", root=out_root)
    buf = ReplayBuffer.from_lerobot_dataset(ds, device="cpu", state_keys=[STATE, CAM], use_drq=False)
    n = len(ds)
    rewards = buf.rewards[:n].tolist()
    dones = buf.dones[:n].tolist()
    assert rewards == [0, 0, 1, 0, 0, 0, 0, 0]
    assert dones == [False, False, True, False, False, False, False, False]
    st = buf.states[STATE][:n, 0].tolist()
    nx = buf.next_states[STATE][:n, 0].tolist()
    # inside an episode next_state = next frame; at the success transition and at the episode end -> itself
    assert nx[0] == st[1] and nx[1] == st[2]
    assert nx[2] == st[2]  # done=True
    assert nx[3:7] == st[4:8]
    assert nx[7] == st[7]  # no-trigger episode end: bootstraps from itself (truncation proxy), not terminal


def test_cli_relabel_demos(src, relabeled, tmp_path):
    from hilreview.cli import main

    _, _, work = relabeled
    (tmp_path / "splits.json").write_text(json.dumps(
        {"episode_split": {"ep0000": "train", "ep0001": "train", "ep0002": "val"}}))
    rc = main(["relabel-demos", "--src-repo-id", "test/src", "--src-root", str(src),
               "--episodes", str(work / "episodes.jsonl"), "--split-file", str(tmp_path / "splits.json"),
               "--scores", str(work / "scores.jsonl"), "--out-repo-id", "test/demo_cli",
               "--out-root", str(tmp_path / "demo_cli"), "--manifest", str(tmp_path / "m.json")])
    assert rc == 0
    ds = LeRobotDataset("test/demo_cli", root=tmp_path / "demo_cli")
    assert len(ds) == 8 and [float(ds[i]["next.reward"]) for i in range(3)] == [0.0, 0.0, 1.0]
