"""SO-101 리더·팔로워 팔의 모터 통신을 읽기 전용으로 확인합니다 (LeRobot 고정 커밋 8c920c42의 FeetechMotorsBus 사용).

모터에 아무것도 쓰지 않습니다:
  - 토크를 켜지 않고, 종료할 때도 토크 끄기 명령을 보내지 않습니다 (disconnect(disable_torque=False)).
  - 캘리브레이션 파일을 쓰거나 읽지 않고, 위치는 원시값(0~4095)으로 봅니다.
확인하는 것 (팔마다):
  1. 고정 이름 포트(/dev/so101_*)가 열리는가, 모터 ID 1~6이 모두 응답하는가, 펌웨어가 같은가
  2. 모터별 현재 위치(원시값), 전압, 온도, 토크 상태, 오류 상태
  3. (선택) 손으로 관절을 움직이며 위치값이 바뀌는지 실시간 확인

실행 (dialout 그룹 필요, lerobot 환경):
  ~/miniforge3/envs/lerobot/bin/python scripts/check_arms.py
단계마다 멈춥니다: Enter = 다음 단계, r = 이 단계 다시, s = 건너뛰기, q = 종료
"""

import os
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARMS = [("리더", "/dev/so101_leader"), ("팔로워", "/dev/so101_follower")]
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LIVE_SECONDS = 50.0


def make_bus(port: str) -> FeetechMotorsBus:
    # same IDs/model as lerobot/robots/so_follower/so_follower.py; norm mode irrelevant (we read raw values)
    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(MOTOR_NAMES)}
    return FeetechMotorsBus(port=port, motors=motors)


_HANGUL_KEYS = {"ㅂ": "q", "ㄱ": "r", "ㄴ": "s"}  # 한글 자판 상태에서 q/r/s 자리 키


def read_cmd(prompt: str, allowed: str) -> str:
    """Enter = "" (다음). allowed 밖의 입력이면 넘어가지 않고 다시 묻습니다."""
    while True:
        c = input(prompt).strip().lower()
        c = _HANGUL_KEYS.get(c, c)
        if c == "" or c in allowed:
            return c
        print(f"  알 수 없는 입력 '{c}'. Enter 또는 {'/'.join(allowed)} 중 하나를 입력하세요.")


def ask(prompt: str = "  Enter = 다음 단계, r = 다시, s = 건너뛰기, q = 종료 > ", allowed: str = "rsq") -> str:
    return read_cmd(prompt, allowed)


def step_connect(label: str, port: str) -> tuple[bool, FeetechMotorsBus | None]:
    real = os.path.realpath(port)
    print(f"  포트 {port} -> {real}")
    if not os.path.exists(port):
        print("  FAIL: 고정 이름 포트가 없습니다. USB 연결과 udev 규칙을 확인하세요.")
        return False, None
    if not os.access(real, os.R_OK | os.W_OK):
        print("  FAIL: 포트 접근 권한이 없습니다. dialout 그룹 적용(재로그인)을 확인하세요.")
        return False, None
    bus = make_bus(port)
    try:
        bus.connect(handshake=True)  # ping ID 1~6 + firmware equality check (read only)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: 연결/핸드셰이크 실패: {e}")
        print("  힌트: 팔 전원 어댑터가 꽂혀 있는지, 다른 프로그램이 포트를 쓰고 있지 않은지 확인하세요.")
        try:
            found = bus.broadcast_ping()
            print(f"  참고: 응답한 모터 ID -> {sorted(found) if found else '없음'}")
        except Exception:  # noqa: BLE001
            pass
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        return False, None
    print(f"  PASS: {label} 팔 모터 ID 1~6 모두 응답, 펌웨어 동일")
    return True, bus


def step_status(bus: FeetechMotorsBus) -> bool:
    print(f"  {'모터':<14}{'ID':>3}{'위치(raw)':>11}{'전압(V)':>9}{'온도(C)':>9}{'토크':>6}{'상태':>6}")
    ok = True
    volts = []
    for name in MOTOR_NAMES:
        r = lambda d: bus.read(d, name, normalize=False)  # noqa: E731
        pos, volt, temp, torque, status = (
            r("Present_Position"),
            r("Present_Voltage") / 10,
            r("Present_Temperature"),
            r("Torque_Enable"),
            r("Status"),
        )
        volts.append(volt)
        flag = "" if status == 0 else "  <- 오류 비트"
        ok &= status == 0
        print(f"  {name:<14}{bus.motors[name].id:>3}{pos:>11}{volt:>9.1f}{temp:>9}{'ON' if torque else 'off':>6}{status:>6}{flag}")
    print(f"  전압 범위 {min(volts):.1f}~{max(volts):.1f} V (리더는 보통 5V대, 팔로워는 어댑터에 따라 7.4V 또는 12V대)")
    print(f"  {'PASS' if ok else 'FAIL'}: 오류 상태 비트 {'없음' if ok else '있음 (위 표 확인)'}")
    return ok


def step_live(bus: FeetechMotorsBus) -> bool:
    print(f"  {LIVE_SECONDS:.0f}초 동안 관절을 하나씩 손으로 천천히 움직여 보세요. 토크가 켜진 모터는 억지로 돌리지 마세요.")
    start = bus.sync_read("Present_Position", normalize=False)
    span = {n: [v, v] for n, v in start.items()}
    end = time.monotonic() + LIVE_SECONDS
    while time.monotonic() < end:
        cur = bus.sync_read("Present_Position", normalize=False)
        for n, v in cur.items():
            span[n][0], span[n][1] = min(span[n][0], v), max(span[n][1], v)
        line = " ".join(f"{n[:8]}={cur[n]:4d}" for n in MOTOR_NAMES)
        print(f"\r  {line}  (남은 {end - time.monotonic():4.1f}s)", end="", flush=True)
        time.sleep(0.1)
    print()
    moved_all = True
    for n in MOTOR_NAMES:
        d = span[n][1] - span[n][0]
        moved = d >= 50
        moved_all &= moved
        print(f"  {n:<14} 변화량 {d:5d} {'(움직임 확인)' if moved else '(거의 안 움직임)'}")
    print(f"  {'PASS' if moved_all else '확인 필요'}: 6개 관절 {'모두' if moved_all else '중 일부만'} 위치값 변화 감지")
    return moved_all


def main() -> int:
    print("SO-101 통신 점검 (읽기 전용: 토크를 켜거나 끄지 않음, 모터에 쓰기 없음)\n")
    results: list[tuple[str, str]] = []
    if ask("시작하려면 Enter (q = 종료) > ", "q") == "q":
        return 1
    quit_all = False
    for label, port in ARMS:
        if quit_all:
            break
        bus = None
        steps = [
            (f"{label} 팔: 연결 + 모터 ID 1~6 응답 + 펌웨어", "connect"),
            (f"{label} 팔: 모터 상태 읽기", "status"),
            (f"{label} 팔: 손으로 움직이며 위치값 변화 확인", "live"),
        ]
        i = 0
        while i < len(steps):
            name, kind = steps[i]
            print(f"\n[{label} {i + 1}/{len(steps)}] {name}")
            if kind == "connect":
                if bus is not None:
                    bus.disconnect(disable_torque=False)
                    bus = None
                ok, bus = step_connect(label, port)
            elif bus is None:
                print("  건너뜀: 연결이 안 되어 있습니다.")
                ok = False
            elif kind == "status":
                ok = step_status(bus)
            else:
                ok = step_live(bus)
            cmd = ask()
            if cmd == "r":
                continue
            results.append((name, "건너뜀" if cmd == "s" else ("PASS" if ok else "FAIL")))
            if cmd == "q":
                quit_all = True
                break
            i += 1
        if bus is not None:
            bus.disconnect(disable_torque=False)

    print("\n결과:")
    for name, st in results:
        print(f"  {st:<6} {name}")
    ok_all = len(results) == 6 and all(st == "PASS" for _, st in results)
    print("-> 전체 PASS" if ok_all else "-> 확인 필요 또는 중간 종료")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
