"""게임패드 버튼·축 번호와 스틱 방향이 LeRobot(고정 커밋 8c920c42)의 하드코딩 값과 맞는지 확인합니다.

근거: lerobot/src/lerobot/teleoperators/gamepad/gamepad_utils.py (GamepadController, pygame 사용)
  버튼 3=성공, 1=실패, 0=다시 녹화, 5=개입(누르는 동안), 6=그리퍼 닫기, 7=그리퍼 열기
  축: delta_x = -axis1, delta_y = -axis0, delta_z = -axis3   (축 2는 사용 안 함)

스틱 점검은 방향마다 따로 합니다. 각 방향에서 확인하는 것:
  - 맞는 축이 움직였는가 (축 번호)
  - 부호가 맞는가 (예: 위로 밀면 음수 -> LeRobot에서 +x/+z)
  - 끝까지 밀었을 때 |값| >= FULL_THRESHOLD 인가 (끝까지 도달)
시작할 때 손을 뗀 상태의 중립값(드리프트)도 잽니다.

로봇은 열지 않습니다. 실행 (lerobot 환경의 pygame 필요):
  ~/miniforge3/envs/lerobot/bin/python scripts/check_gamepad.py
단계마다 멈춥니다: Enter = 다음 단계, r = 이 단계 다시, q = 종료
"""

import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

TIMEOUT_S = 15.0
DETECT_THRESHOLD = 0.6  # 이 값을 넘으면 "움직였다"고 봄
FULL_THRESHOLD = 0.9  # 끝까지 밀었을 때 이 값 이상이어야 함
RELEASE_THRESHOLD = 0.2  # 이 값 아래로 돌아오면 "놓았다"고 봄
DRIFT_LIMIT = 0.1  # 손을 뗐을 때 허용하는 중립 오차
HOLD_MAX_S = 5.0  # 끝까지 민 뒤 놓기를 기다리는 최대 시간

# 버튼: (안내, 기대 번호, LeRobot 의미)
BUTTONS = [
    ("Y 버튼을 누르세요", 3, "에피소드 성공"),
    ("A 버튼을 누르세요", 1, "에피소드 실패"),
    ("X 버튼을 누르세요", 0, "다시 녹화"),
    ("RB (오른쪽 위 범퍼)를 누르세요", 5, "개입 (누르는 동안)"),
    ("LT (왼쪽 아래 트리거)를 누르세요", 6, "그리퍼 닫기"),
    ("RT (오른쪽 아래 트리거)를 누르세요", 7, "그리퍼 열기"),
]

# 스틱: (안내, 기대 축, 기대 부호, LeRobot 의미)
STICKS = [
    ("왼쪽 스틱을 왼쪽 끝까지 밀었다가 놓으세요", 0, -1, "+y 이동"),
    ("왼쪽 스틱을 오른쪽 끝까지 밀었다가 놓으세요", 0, +1, "-y 이동"),
    ("왼쪽 스틱을 위쪽 끝까지 밀었다가 놓으세요", 1, -1, "+x 이동"),
    ("왼쪽 스틱을 아래쪽 끝까지 밀었다가 놓으세요", 1, +1, "-x 이동"),
    ("오른쪽 스틱을 왼쪽 끝까지 밀었다가 놓으세요", 2, -1, "LeRobot 사용 안 함 (참고)"),
    ("오른쪽 스틱을 오른쪽 끝까지 밀었다가 놓으세요", 2, +1, "LeRobot 사용 안 함 (참고)"),
    ("오른쪽 스틱을 위쪽 끝까지 밀었다가 놓으세요", 3, -1, "+z 이동 (위)"),
    ("오른쪽 스틱을 아래쪽 끝까지 밀었다가 놓으세요", 3, +1, "-z 이동 (아래)"),
]


def pump_axes(j: "pygame.joystick.JoystickType") -> list[float]:
    pygame.event.pump()
    return [j.get_axis(a) for a in range(j.get_numaxes())]


def check_center(j: "pygame.joystick.JoystickType") -> bool:
    """Sample all axes for 1 s with hands off; report the worst offset per axis."""
    worst = [0.0] * j.get_numaxes()
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        for a, v in enumerate(pump_axes(j)):
            if abs(v) > abs(worst[a]):
                worst[a] = v
        time.sleep(0.01)
    ok = all(abs(v) <= DRIFT_LIMIT for v in worst)
    vals = ", ".join(f"축{a}={v:+.3f}" for a, v in enumerate(worst))
    print(f"  {'PASS' if ok else 'FAIL'} 중립값 (허용 ±{DRIFT_LIMIT}): {vals}")
    return ok


def check_button(expected: int) -> tuple[str, str]:
    pygame.event.clear()
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                status = "PASS" if e.button == expected else "FAIL"
                return status, f"기대 버튼 {expected}, 받은 값 {e.button}"
        time.sleep(0.01)
    return "시간 초과", f"기대 버튼 {expected}, 입력 없음"


def check_stick(j: "pygame.joystick.JoystickType", axis: int, sign: int) -> tuple[str, str]:
    # 1) wait until any axis leaves the center
    pygame.event.clear()
    deadline = time.monotonic() + TIMEOUT_S
    moved = None
    while time.monotonic() < deadline and moved is None:
        for a, v in enumerate(pump_axes(j)):
            if abs(v) > DETECT_THRESHOLD:
                moved = a
                break
        time.sleep(0.005)
    if moved is None:
        return "시간 초과", f"기대 축 {axis}, 입력 없음"

    # 2) track the peak on the moved axis until the stick is released
    peak = 0.0
    end = time.monotonic() + HOLD_MAX_S
    while time.monotonic() < end:
        v = pump_axes(j)[moved]
        if abs(v) > abs(peak):
            peak = v
        if abs(v) < RELEASE_THRESHOLD:
            break
        time.sleep(0.005)

    got_sign = 1 if peak > 0 else -1
    detail = f"기대 축 {axis} 부호 {'+' if sign > 0 else '-'} | 받은 값 축 {moved}, 최댓값 {peak:+.3f}"
    if moved != axis:
        return "FAIL", detail + " (다른 축)"
    if got_sign != sign:
        return "FAIL", detail + " (방향 반대)"
    if abs(peak) < FULL_THRESHOLD:
        return "FAIL", detail + f" (끝까지 도달 안 함, 기준 {FULL_THRESHOLD})"
    return "PASS", detail


_HANGUL_KEYS = {"ㅂ": "q", "ㄱ": "r", "ㄴ": "s"}  # 한글 자판 상태에서 q/r/s 자리 키


def read_cmd(prompt: str, allowed: str) -> str:
    """Enter = "" (다음). allowed 밖의 입력이면 넘어가지 않고 다시 묻습니다."""
    while True:
        c = input(prompt).strip().lower()
        c = _HANGUL_KEYS.get(c, c)
        if c == "" or c in allowed:
            return c
        print(f"  알 수 없는 입력 '{c}'. Enter 또는 {'/'.join(allowed)} 중 하나를 입력하세요.")


def ask_next() -> str:
    return read_cmd("  Enter = 다음 단계, r = 다시, q = 종료 > ", "rq")


def main() -> int:
    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        print("FAIL: 게임패드를 찾지 못했습니다. USB 연결을 확인하세요.")
        return 1
    if n > 1:
        print(f"주의: 게임패드가 {n}개 연결되어 있습니다. LeRobot은 0번만 씁니다.")
    j = pygame.joystick.Joystick(0)
    j.init()
    print(f"0번 게임패드: {j.get_name()} | 버튼 {j.get_numbuttons()}개, 축 {j.get_numaxes()}개")
    print("F310은 뒷면 스위치 D, 앞면 MODE 불 꺼짐 상태여야 합니다.")
    print(f"각 단계는 시작 후 {TIMEOUT_S:.0f}초 안에 입력하세요.")
    print("단계가 끝날 때마다 멈춥니다: Enter = 다음 단계, r = 이 단계 다시, q = 종료\n")

    # (이름, 실행 함수)
    steps = [("중립값: 스틱에서 손을 떼고 기다리세요 (1초)", lambda: check_center(j))]
    steps += [(f"{m}: {p}", (lambda e=e: check_button(e))) for p, e, m in BUTTONS]
    steps += [(f"{m}: {p}", (lambda a=a, s=s: check_stick(j, a, s))) for p, a, s, m in STICKS]

    if read_cmd("시작하려면 Enter (q = 종료) > ", "q") == "q":
        return 1

    results: list[tuple[str, bool]] = []
    i = 0
    while i < len(steps):
        name, fn = steps[i]
        print(f"\n[{i + 1}/{len(steps)}] {name} ...", end=" " if i else "\n", flush=True)
        out = fn()
        if isinstance(out, bool):  # center check prints its own line
            ok = out
        else:
            status, detail = out
            ok = status == "PASS"
            print(f"{status} ({detail})")
        time.sleep(0.5)
        pygame.event.clear()

        cmd = ask_next()
        if cmd == "r":
            continue
        results.append((name, ok))
        if cmd == "q":
            break
        i += 1

    j.quit()
    pygame.quit()
    done = len(results) == len(steps)
    passed = sum(ok for _, ok in results)
    ok_all = done and passed == len(steps)
    tail = "전체 PASS" if ok_all else ("확인 필요" if done else "중간에 종료")
    print(f"\n결과: {passed}/{len(steps)} 통과 -> {tail}")
    for name, ok in results:
        if not ok:
            print(f"  실패: {name}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
