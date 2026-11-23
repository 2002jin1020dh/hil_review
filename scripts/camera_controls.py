#!/usr/bin/env python
"""실험 카메라 노출·화이트밸런스 고정 (D-39). 로봇은 쓰지 않습니다. lerobot 환경 (opencv) 필요.

  show                       두 카메라의 모든 제어값 보기
  lock  --out <파일.json>    지금 장면에서 자동 노출/화이트밸런스가 만드는 밝기·색에 가장 가까운 수동값을 찾아
                             적용하고 파일로 저장 (물체·용기·조명을 실험 상태로 둔 뒤 한 번)
  apply --file <파일.json> [--hold]
                             저장한 값을 다시 적용하고 확인. --hold: 세션 내내 다른 터미널에서 켜 두기 (C270은 영상이
                             시작될 때 이득을 되돌리므로, LeRobot이 카메라를 연 뒤에도 값이 유지되게 주기적으로 재적용)

공통으로: 전원 주파수 60Hz(한국), 노출에 따른 프레임 속도 낮추기 끔, 노출 상한 333(=33.3ms, 30fps 유지).
  ~/miniforge3/envs/lerobot/bin/python scripts/camera_controls.py lock --out configs/cameras/session.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hilreview.robot.v4l2_controls import (
    AUTO_EXPOSURE,
    EXPOSURE_ABS,
    EXPOSURE_DYN_FPS,
    POWER_LINE,
    WB_AUTO,
    WB_TEMP,
    Camera,
)

CAMERAS = {"front": "/dev/cam_front", "wrist": "/dev/cam_wrist"}
EXPOSURE_MAX_30FPS = 333  # units of 100 us
COMMON = {POWER_LINE: 2, EXPOSURE_DYN_FPS: 0}


def grab(cap: cv2.VideoCapture, n: int = 8, discard: int = 6) -> np.ndarray:
    for _ in range(discard):
        cap.read()
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            frames.append(f.astype(np.float32))
    if not frames:
        raise RuntimeError("카메라에서 프레임을 받지 못했습니다")
    return np.mean(frames, axis=0)


def stats(img: np.ndarray) -> dict:
    b, g, r = (float(img[..., i].mean()) for i in range(3))
    return {"luma": 0.114 * b + 0.587 * g + 0.299 * r, "r_g": r / max(g, 1e-6), "b_g": b / max(g, 1e-6)}


def measure_fps(cap: cv2.VideoCapture, n: int = 30) -> float:
    cap.read()
    t = time.perf_counter()
    for _ in range(n):
        cap.read()
    return n / (time.perf_counter() - t)


def lock_one(name: str, path: str, settle_s: float) -> dict:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    with Camera(path) as cam:
        ctrls = cam.controls()
        cam.set_by_name({**{k: v for k, v in COMMON.items() if k in ctrls}, AUTO_EXPOSURE: 3, WB_AUTO: 1})
        time.sleep(settle_s)
        target = stats(grab(cap, n=15, discard=30))
        # manual exposure: exposure at the 30 fps cap, then match brightness with gain if the camera has one,
        # otherwise bisection on exposure. If manual cannot reach 80% of the auto brightness at 30 fps
        # (camera uses hidden auto gain), keep AUTO exposure and only lock white balance (recorded in the file).
        cam.set_by_name({AUTO_EXPOSURE: 1, WB_AUTO: 0})
        cap_e = min(EXPOSURE_MAX_30FPS, ctrls[EXPOSURE_ABS].maximum)
        knob, lo, hi = (("gain", ctrls["gain"].minimum, ctrls["gain"].maximum) if "gain" in ctrls
                        else (EXPOSURE_ABS, max(1, ctrls[EXPOSURE_ABS].minimum), cap_e))
        if knob == "gain":
            cam.set_by_name({EXPOSURE_ABS: cap_e})
        best = None
        for _ in range(12):
            mid = (lo + hi) // 2
            cam.set_by_name({knob: mid})
            s = stats(grab(cap))
            if best is None or abs(s["luma"] - target["luma"]) < abs(best[1]["luma"] - target["luma"]):
                best = (mid, s)
            if s["luma"] < target["luma"]:
                lo = mid + 1
            else:
                hi = mid - 1
            if lo > hi:
                break
        exposure_mode = "manual"
        if best[1]["luma"] < 0.8 * target["luma"]:
            exposure_mode = "auto (수동으로 30fps에서 자동 밝기의 80%에 못 미침)"
            cam.set_by_name({AUTO_EXPOSURE: 3})
            time.sleep(settle_s)
        else:
            cam.set_by_name({knob: best[0]})
        exposure = (knob, best[0]) if exposure_mode == "manual" else None
        # white balance: grid over temperature, closest chromaticity to the auto image
        wt = ctrls[WB_TEMP]
        best_wb = None
        for temp in range(max(wt.minimum, 2800), min(wt.maximum, 6500) + 1, 200):
            temp -= temp % max(wt.step, 1)
            cam.set_by_name({WB_TEMP: temp})
            s = stats(grab(cap))
            err = abs(s["r_g"] - target["r_g"]) + abs(s["b_g"] - target["b_g"])
            if best_wb is None or err < best_wb[1]:
                best_wb = (temp, err)
        cam.set_by_name({WB_TEMP: best_wb[0]})
        final = stats(grab(cap, n=15))
        fps = measure_fps(cap)
        keys = [*[c for c in COMMON if c in ctrls], AUTO_EXPOSURE, WB_AUTO, WB_TEMP]
        if exposure_mode == "manual":
            keys += [EXPOSURE_ABS] + (["gain"] if "gain" in ctrls else [])
        values = {k: cam.controls()[k].value for k in keys}
    cap.release()
    expo_txt = f"수동 {exposure[0]}={exposure[1]}" if exposure else "자동 노출 유지"
    print(f"[{name}] 자동 밝기 {target['luma']:.1f} -> {expo_txt}, 밝기 {final['luma']:.1f} | "
          f"색 R/G {target['r_g']:.3f}->{final['r_g']:.3f}, B/G {target['b_g']:.3f}->{final['b_g']:.3f} "
          f"(색온도 {best_wb[0]}K) | {fps:.1f} fps")
    return {"path": path, "exposure_mode": exposure_mode, "values": values, "auto_target": target, "locked": final,
            "fps": fps}


ORDER = (POWER_LINE, EXPOSURE_DYN_FPS, AUTO_EXPOSURE, WB_AUTO, EXPOSURE_ABS, "gain", WB_TEMP)


def _set_all(data: dict) -> dict[str, bool]:
    ok = {}
    for name, d in data["cameras"].items():
        order = [k for k in ORDER if k in d["values"]]
        with Camera(d["path"]) as cam:
            got = cam.set_by_name({k: d["values"][k] for k in order})
        ok[name] = got == {k: d["values"][k] for k in order}
    return ok


def apply(file: Path, hold: bool, interval: float) -> int:
    """C270 quirk (measured 2026-11-23): starting a stream resets the sensor gain while the driver still reports the
    set value, so values must be (re)applied WHILE the camera streams. LeRobot opens the cameras itself, hence --hold:
    keep re-applying the same values every few seconds during the whole session."""
    data = json.loads(file.read_text())
    caps = {}
    for name, d in data["cameras"].items():  # open the streams first (as LeRobot will), then set
        cap = cv2.VideoCapture(d["path"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.read()
        caps[name] = cap
    ok = _set_all(data)
    for name, d in data["cameras"].items():
        s = stats(grab(caps[name], n=15, discard=20))
        fps = measure_fps(caps[name])
        dl = s["luma"] - d["locked"]["luma"]
        print(f"[{name}] 적용 {'OK' if ok[name] else 'FAIL'} ({d.get('exposure_mode', 'manual')}) | 밝기 {s['luma']:.1f} "
              f"(저장 때 {d['locked']['luma']:.1f}, 차이 {dl:+.1f}) | {fps:.1f} fps")
        caps[name].release()
    print("밝기 차이가 크면(약 ±10 이상) 조명이나 장면이 저장 때와 다릅니다.")
    if not hold:
        return 0 if all(ok.values()) else 1
    print(f"--hold: {interval:.0f}초마다 같은 값을 다시 적용합니다 (LeRobot이 카메라를 다시 열어도 유지). 끝내려면 Ctrl+C")
    try:
        while True:
            time.sleep(interval)
            try:
                _set_all(data)
            except OSError as exc:  # camera unplugged / busy for a moment
                print(f"재적용 실패 (계속 시도): {exc}")
    except KeyboardInterrupt:
        print("중지")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    lk = sub.add_parser("lock")
    lk.add_argument("--out", required=True)
    lk.add_argument("--settle", type=float, default=3.0, help="자동 노출이 안정될 때까지 기다리는 초")
    ap_ = sub.add_parser("apply")
    ap_.add_argument("--file", required=True)
    ap_.add_argument("--hold", action="store_true", help="세션 내내 주기적으로 다시 적용 (녹화·학습 중 켜 둠)")
    ap_.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "show":
        for name, path in CAMERAS.items():
            with Camera(path) as cam:
                print(f"== {name} ({path})")
                for n, c in cam.controls().items():
                    print(f"  {n:<30} {c.value!s:>6}  [{c.minimum},{c.maximum}]{'  (비활성)' if c.inactive else ''}")
        return 0
    if a.cmd == "lock":
        res = {name: lock_one(name, path, a.settle) for name, path in CAMERAS.items()}
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "cameras": res},
                                  indent=2, ensure_ascii=False))
        print(f"저장: {out}  (세션 시작 때: camera_controls.py apply --file {out})")
        return 0
    return apply(Path(a.file), a.hold, a.interval)


if __name__ == "__main__":
    sys.exit(main())
