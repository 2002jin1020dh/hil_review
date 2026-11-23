"""front·wrist 카메라 실시간 보기 + 캡처 (로봇은 열지 않음).

실행 (lerobot 환경의 pygame, opencv 사용):
  ~/miniforge3/envs/lerobot/bin/python scripts/camera_view.py [--out ~/research/captures]
키:
  Space 또는 c : 두 카메라 화면을 동시에 저장 (원본 해상도, JPG)
  q 또는 Esc   : 종료
저장 이름: <out>/<날짜_시각>_<번호>_front.jpg, ..._wrist.jpg
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import cv2  # noqa: E402
import pygame  # noqa: E402

CAMERAS = [("front", "/dev/cam_front"), ("wrist", "/dev/cam_wrist")]
W, H = 640, 480
BAR = 36


class Grabber(threading.Thread):
    """Keeps the latest frame of one camera so the UI never blocks on a slow read."""

    def __init__(self, name: str, path: str):
        super().__init__(daemon=True)
        self.name, self.path = name, path
        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.frame = None
        self.fps = 0.0
        self.running = self.cap.isOpened()
        self.lock = threading.Lock()

    def run(self) -> None:
        n, t0 = 0, time.monotonic()
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
            n += 1
            if time.monotonic() - t0 >= 1.0:
                self.fps, n, t0 = n / (time.monotonic() - t0), 0, time.monotonic()

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False
        self.join(timeout=1.0)
        self.cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "research" / "captures"))
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    grabbers = [Grabber(n, p) for n, p in CAMERAS]
    for g in grabbers:
        if not g.running:
            print(f"열 수 없음: {g.name} ({g.path})")
        g.start()

    pygame.init()
    screen = pygame.display.set_mode((W * len(grabbers), H + BAR))
    pygame.display.set_caption("hil-review 카메라 (Space/c = 캡처, q/Esc = 종료)")
    font = pygame.font.SysFont("notosanscjkkr,notoserifcjkkr,nanumgothic,unifont,dejavusans", 18)
    clock = pygame.time.Clock()
    count, msg, msg_until = 0, "", 0.0
    print(f"캡처 저장 폴더: {out}")

    running = True
    while running:
        frames = [g.latest() for g in grabbers]
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key in (pygame.K_SPACE, pygame.K_c):
                    count += 1
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved = []
                    for g, f in zip(grabbers, frames):
                        if f is not None:
                            p = out / f"{stamp}_{count:02d}_{g.name}.jpg"
                            cv2.imwrite(str(p), f)
                            saved.append(p.name)
                    msg, msg_until = f"저장 #{count}: {', '.join(saved)}", time.monotonic() + 3
                    print(msg)

        screen.fill((20, 20, 20))
        for i, (g, f) in enumerate(zip(grabbers, frames)):
            if f is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB)
            screen.blit(pygame.image.frombuffer(rgb.tobytes(), (W, H), "RGB"), (i * W, 0))
            screen.blit(font.render(f"{g.name}  {g.fps:4.1f} fps", True, (255, 255, 0)), (i * W + 8, 6))
        status = msg if time.monotonic() < msg_until else f"Space/c = 캡처 ({count}장)   q/Esc = 종료   저장: {out}"
        screen.blit(font.render(status, True, (230, 230, 230)), (8, H + 8))
        pygame.display.flip()
        clock.tick(30)

    for g in grabbers:
        g.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
