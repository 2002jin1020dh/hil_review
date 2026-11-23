#!/usr/bin/env python
"""오프라인 파이프라인 시운전 (SYNTHETIC 데이터, 실제 LeRobot 8c920c42 + GPU). 로봇·카메라는 쓰지 않습니다.

실제 실험과 같은 순서로 CLI를 실행해 끝까지 도는지, 그리고 아래 두 가지를 확인합니다.
  1. 학습-실행 입력 불일치 (PROJECT_STATUS 문제 1): lerobot-train은 이미지를 MEAN_STD로 정규화하지만,
     실행 중 RewardClassifierProcessorStep은 정규화 없이 분류기를 부릅니다. 기본 설정(MEAN_STD)과
     normalization_mapping VISUAL=IDENTITY 설정으로 분류기를 하나씩 학습해, 두 입력 방식의 점수를 비교합니다.
  2. hilreview score --input-mode runtime 이 실제 RewardClassifierProcessorStep 판정과 같은지.
마지막으로 D-29 재라벨(score-frames -> relabel-demos -> ReplayBuffer)까지 실행합니다.

실행 (lerobot 환경, 첫 실행 때 lerobot/resnet10 가중치를 Hugging Face에서 내려받음):
  ~/miniforge3/envs/lerobot/bin/python scripts/dry_run_offline.py --work ~/research/dry_run [--steps 400]
결과: <work>/summary.json 과 화면 표. 합성 데이터 결과는 "프로그램이 도는지"만 말해 줍니다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BIN = Path(sys.executable).parent
H = W = 128
FPS = 10
N_FRAMES = 40
CAMS = ("observation.images.front", "observation.images.wrist")
CONTAINER = (70, 110, 70, 110)  # rows, cols of the container square in the front image
TASK = "put the object into the container"


def run(cmd: list[str], log: Path) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"실패 (exit {r.returncode}): 로그 {log}")


# ------------------------------------------------------------------------------------------------
# 1. Fake recording in the exact gym_manipulator record-mode layout
# ------------------------------------------------------------------------------------------------


def _front(pos, rng):
    img = np.full((3, H, W), 0.15, np.float32) + rng.normal(0, 0.02, (3, H, W)).astype(np.float32)
    r0, r1, c0, c1 = CONTAINER
    img[:, r0:r1, c0:c1] = 0.35
    img[:, r0 + 3:r1 - 3, c0 + 3:c1 - 3] = 0.25
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (yy - pos[0]) ** 2 + (xx - pos[1]) ** 2 <= 8 ** 2
    img[:, disc] = 0.95
    return np.clip(img, 0, 1)


def _wrist(progress, rng):
    img = np.full((3, H, W), 0.2, np.float32) + rng.normal(0, 0.03, (3, H, W)).astype(np.float32)
    s = int(10 + 40 * progress)
    img[:, H // 2 - s // 2:H // 2 + s // 2, W // 2 - s // 2:W // 2 + s // 2] = 0.9
    return np.clip(img, 0, 1)


def make_dataset(root: Path, n_eps: int, seed: int) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(seed)
    features = {
        "action": {"dtype": "float32", "shape": (4,), "names": ["delta_x", "delta_y", "delta_z", "gripper"]},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.discrete_penalty": {"dtype": "float32", "shape": (1,), "names": ["discrete_penalty"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        **{c: {"dtype": "video", "shape": (3, H, W), "names": ["channels", "height", "width"]} for c in CAMS},
    }
    ds = LeRobotDataset.create("dry/raw", fps=FPS, features=features, root=root, use_videos=True)
    modes = ["success_attempt", "success_attempt", "near_miss", "clear_failure"]
    meta = {}
    for ep in range(n_eps):
        mode = modes[ep % len(modes)]
        start = np.array([20.0 + rng.uniform(-5, 5), 20.0 + rng.uniform(-5, 5)])
        target = {"success_attempt": np.array([90.0, 90.0]), "near_miss": np.array([90.0, 58.0]),
                  "clear_failure": np.array([35.0, 40.0])}[mode]
        onset = int(rng.integers(20, 30)) if mode == "success_attempt" else None
        arrive = onset if onset is not None else int(rng.integers(20, 30))
        for f in range(N_FRAMES):
            prog = min(1.0, f / arrive)
            pos = start + (target - start) * prog + rng.normal(0, 0.7, 2)
            success = onset is not None and f >= onset
            ds.add_frame({
                "action": rng.normal(0, 0.1, 4).astype(np.float32),
                "next.reward": np.array([1.0 if success else 0.0], np.float32),  # human Y button
                "next.done": np.array([f == N_FRAMES - 1], bool),
                "complementary_info.discrete_penalty": np.array([0.0], np.float32),
                "observation.state": rng.normal(0, 1, 6).astype(np.float32),
                CAMS[0]: _front(pos, rng), CAMS[1]: _wrist(prog, rng), "task": TASK,
            })
        ds.save_episode()
        meta[str(ep)] = {"session_id": f"s{ep % 3}", "collection_mode": mode, "onset": onset}
    ds.finalize()
    return meta


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------


def truth_target(meta: dict, ep_idx: int, frame: int) -> int:
    onset = meta[str(ep_idx)]["onset"]
    return int(onset is not None and frame >= onset)


def find_checkpoint(out_dir: Path) -> Path:
    cands = sorted(out_dir.glob("checkpoints/*/pretrained_model"))
    if not cands:
        raise SystemExit(f"체크포인트가 없습니다: {out_dir}")
    return cands[-1]


def accuracy(scores_jsonl: Path, sample_truth: dict[str, int]) -> dict:
    rows = [json.loads(line) for line in scores_jsonl.read_text().splitlines()]
    tp = fp = tn = fn = 0
    ps = []
    for r in rows:
        y, p = sample_truth[r["sample_id"]], r["p_success"]
        ps.append(p)
        pred = int(p > 0.5)
        tp += pred and y
        fp += pred and not y
        tn += (not pred) and (not y)
        fn += (not pred) and y
    n = tp + fp + tn + fn
    return {"n": n, "accuracy": (tp + tn) / n, "fpr": fp / max(1, fp + tn), "fnr": fn / max(1, fn + tp),
            "mean_p": float(np.mean(ps)), "frac_pred_success": (tp + fp) / n}


def processor_step_agreement(ckpt: Path, raw_root: Path, frames: list[tuple[int, float]]) -> dict:
    """Run the REAL RewardClassifierProcessorStep on dataset frames; compare with runtime-mode p > 0.5."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.hil_processor import RewardClassifierProcessorStep
    from lerobot.lerobot_types import TransitionKey

    step = RewardClassifierProcessorStep(pretrained_path=str(ckpt), device="cuda", success_threshold=0.5)
    ds = LeRobotDataset("dry/raw", root=raw_root)
    agree = 0
    for idx, p in frames:
        item = ds[idx]
        obs = {c: item[c].unsqueeze(0).to("cuda") for c in CAMS}
        tr = {TransitionKey.OBSERVATION: obs, TransitionKey.REWARD: 0.0, TransitionKey.DONE: False,
              TransitionKey.INFO: {}, TransitionKey.COMPLEMENTARY_DATA: {}}
        out = step(tr)
        agree += int((float(out[TransitionKey.REWARD]) == 1.0) == (p > 0.5))
    return {"n": len(frames), "agree": agree}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="기존 작업 폴더를 지우지 않음")
    a = ap.parse_args()
    work = Path(a.work).expanduser()
    if work.exists() and not a.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "commands.log"
    hil, lt = BIN / "hilreview", BIN / "lerobot-train"
    summary: dict = {"data_kind": "synthetic", "note": "합성 데이터: 프로그램 동작 확인용, 실제 성능 증거 아님"}

    print("\n[1/7] 녹화 형식 가짜 데이터셋 만들기")
    meta = make_dataset(work / "raw", a.episodes, a.seed)
    (work / "episode_meta.json").write_text(json.dumps(meta, indent=1))

    print("\n[2/7] 키프레임 내보내기 + 데이터 나누기")
    run([hil, "export-keyframes", "--repo-id", "dry/raw", "--root", work / "raw", "--out", work / "kf",
         "--stride", "5", "--episode-meta", work / "episode_meta.json"], log)
    run([hil, "split", "--episodes", work / "kf/episodes.jsonl", "--out", work / "splits.json"], log)
    split = json.loads((work / "splits.json").read_text())["episode_split"]
    kfs = [json.loads(line) for line in (work / "kf/keyframes.jsonl").read_text().splitlines()]
    ep_index = {e: int(e[2:]) for e in split}
    sample_truth = {k["sample_id"]: truth_target(meta, ep_index[k["episode_id"]], k["frame_index"]) for k in kfs}
    summary["split_counts"] = json.loads((work / "splits.json").read_text())["counts"]

    print("\n[3/7] 조건 라벨 파일 (합성: 정답 라벨 = 조건 A 흉내) -> 분류기 학습 데이터셋 (train만)")
    rows = [{"condition": "A", "replicate": 0, "sample_id": k["sample_id"], "episode_id": k["episode_id"],
             "frame_index": k["frame_index"], "label": "success" if sample_truth[k["sample_id"]] else "failure",
             "target": sample_truth[k["sample_id"]], "source": "synthetic_truth"}
            for k in kfs if split[k["episode_id"]] in ("train", "val")]
    (work / "labels_A.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run([hil, "build-reward-dataset", "--src-repo-id", "dry/raw", "--src-root", work / "raw",
         "--episodes", work / "kf/episodes.jsonl", "--labels", work / "labels_A.jsonl",
         "--out-repo-id", "dry/rew_A", "--out-root", work / "rew_A", "--split-file", work / "splits.json"], log)

    print("\n[4/7] 보상 분류기 학습 2개 (GPU): 기본 MEAN_STD / VISUAL=IDENTITY")
    ckpts = {}
    for name, norm in (("meanstd", "MEAN_STD"), ("identity", "IDENTITY")):
        out = work / f"clf_{name}"
        cfg = work / f"clf_{name}.json"
        run([hil, "classifier-config", "--dataset-repo-id", "dry/rew_A", "--dataset-root", work / "rew_A",
             "--output-dir", out, "--job-name", f"dry_{name}", "--out", cfg, "--steps", a.steps,
             "--normalization", norm, "--num-workers", 2], log)
        run([lt, f"--config_path={cfg}"], log)
        ckpts[name] = find_checkpoint(out)

    print("\n[5/7] val+audit 키프레임 채점: 분류기 2개 x 입력 방식 2개")
    summary["keyframe_scores"] = {}
    for name, ck in ckpts.items():
        for mode in ("runtime", "train"):
            out = work / f"scores_{name}_{mode}.jsonl"
            run([hil, "score", "--classifier", ck, "--repo-id", "dry/raw", "--root", work / "raw",
                 "--episodes", work / "kf/episodes.jsonl", "--keyframes", work / "kf/keyframes.jsonl",
                 "--split-file", work / "splits.json", "--splits", "val,audit", "--out", out,
                 "--condition", "A", "--replicate", "0", "--input-mode", mode], log)
            summary["keyframe_scores"][f"{name}/{mode}"] = accuracy(out, sample_truth)

    print("\n[6/7] runtime 채점 == 실제 RewardClassifierProcessorStep 판정?")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from hilreview.robot.lerobot_io import _episode_bounds

    bounds = _episode_bounds(LeRobotDataset("dry/raw", root=work / "raw"))
    summary["processor_step_agreement"] = {}
    for name, ck in ckpts.items():
        sc = [json.loads(line) for line in (work / f"scores_{name}_runtime.jsonl").read_text().splitlines()][:60]
        kf_by_id = {k["sample_id"]: k for k in kfs}
        frames = [(bounds[ep_index[kf_by_id[r["sample_id"]]["episode_id"]]][0] + kf_by_id[r["sample_id"]]["frame_index"],
                   r["p_success"]) for r in sc]
        summary["processor_step_agreement"][name] = processor_step_agreement(ck, work / "raw", frames)

    print("\n[7/7] D-29 재라벨: score-frames -> relabel-demos -> ReplayBuffer (IDENTITY 분류기)")
    run([hil, "score-frames", "--classifier", ckpts["identity"], "--repo-id", "dry/raw", "--root", work / "raw",
         "--episodes", work / "kf/episodes.jsonl", "--split-file", work / "splits.json",
         "--out", work / "frame_scores_A.jsonl", "--condition", "A", "--replicate", "0"], log)
    run([hil, "relabel-demos", "--src-repo-id", "dry/raw", "--src-root", work / "raw",
         "--episodes", work / "kf/episodes.jsonl", "--split-file", work / "splits.json",
         "--scores", work / "frame_scores_A.jsonl", "--out-repo-id", "dry/demo_A", "--out-root", work / "demo_A",
         "--manifest", work / "demo_A_manifest.json"], log)
    man = json.loads((work / "demo_A_manifest.json").read_text())["episodes"]
    rel = []
    for m in man:
        onset = meta[str(ep_index[m["episode_id"]])]["onset"]
        rel.append({"episode_id": m["episode_id"], "true_onset": onset, "trigger_frame": m["trigger_frame"],
                    "human_button_frame": m["human_button_frame"],
                    "trigger_minus_expected": None if m["trigger_frame"] is None else m["trigger_frame"] - (onset - 1)})
    from lerobot.rl.buffer import ReplayBuffer

    demo = LeRobotDataset("dry/demo_A", root=work / "demo_A")
    buf = ReplayBuffer.from_lerobot_dataset(demo, device="cpu", state_keys=["observation.state", *CAMS], use_drq=False)
    summary["demo_relabel"] = {"episodes": rel, "buffer_size": len(demo),
                               "buffer_reward_sum": float(buf.rewards[:len(demo)].sum()),
                               "buffer_done_sum": int(buf.dones[:len(demo)].sum())}

    (work / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("\n===== 요약 (합성 데이터) =====")
    print(f"나누기: {summary['split_counts']}")
    print(f"{'분류기/입력':<20}{'정확도':>8}{'FPR':>8}{'FNR':>8}{'성공예측비율':>14}{'평균 p':>9}")
    for k, v in summary["keyframe_scores"].items():
        print(f"{k:<20}{v['accuracy']:>8.3f}{v['fpr']:>8.3f}{v['fnr']:>8.3f}{v['frac_pred_success']:>14.3f}{v['mean_p']:>9.3f}")
    for k, v in summary["processor_step_agreement"].items():
        print(f"ProcessorStep 판정 == runtime 채점 ({k}): {v['agree']}/{v['n']}")
    d = summary["demo_relabel"]
    print(f"재라벨 시연 에피소드 {len(rel)}개: 판정 위치 - (onset-1) = {[r['trigger_minus_expected'] for r in rel]}")
    print(f"버퍼: 전이 {d['buffer_size']}개, 보상 합 {d['buffer_reward_sum']}, done 합 {d['buffer_done_sum']}")
    print(f"전체 결과: {work / 'summary.json'}  명령 로그: {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
