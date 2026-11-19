"""Interactive episode review tool (terminal + filmstrip image). Produces ReviewRecords with MEASURED time.

Usage is through ``hilreview review ...`` (see README). One episode at a time: a filmstrip of its keyframes is
written to disk (and optionally opened in an image viewer); the reviewer types one command in the terminal.
Time per episode = monotonic clock from "filmstrip ready" to "valid command entered". Each finished episode is
appended to the output JSONL immediately (crash-safe); re-running resumes with unfinished episodes.

Blind mode (purpose = audit / audit_repeat) refuses to receive VLM suggestions at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from hilreview.data.io import append_jsonl, read_jsonl
from hilreview.data.schema import DataKind, Episode, Keyframe, Label, ReviewRecord, VLMSuggestion
from hilreview.utils.provenance import utc_now_iso
from hilreview.utils.rng import make_rng

HELP = """명령어 (하나 입력하고 Enter):
  Enter 또는 y   화면의 VLM 제안을 그대로 받아들이기 (보조 검토에서만)
  <숫자>         첫 번째 '성공' 키프레임 번호: 그 앞은 실패, 그 번호부터 끝까지는 성공
  n              이 에피소드에는 성공이 없음 (전부 실패)
  e <글자들>     키프레임마다 글자 하나씩: S 성공, F 실패, A 애매함, I 무효
                 예) 키프레임이 8개면:  e FFFFFSSS
  x              에피소드 전체가 쓸 수 없음 (전부 무효)
  s              나중에 하기 (저장 안 함, 다음에 다시 나옴)
  q              끝내기 (지금까지 한 것은 이미 저장됨)
  ?              이 도움말 보기
"""

_LETTER = {"S": Label.SUCCESS, "F": Label.FAILURE, "A": Label.AMBIGUOUS, "I": Label.INVALID}
_SHORT = {Label.SUCCESS: "S", Label.FAILURE: "F", Label.AMBIGUOUS: "A", Label.INVALID: "I", None: "?"}


def parse_command(cmd: str, n: int, vlm_labels: Sequence[Label | None] | None) -> tuple[str, list[Label] | None, str]:
    """Return (action, labels, message); action in {label, skip, quit, help, error}."""
    c = cmd.strip()
    low = c.lower()
    if low in ("q", "quit"):
        return "quit", None, ""
    if low in ("s", "skip"):
        return "skip", None, ""
    if low in ("?", "h", "help"):
        return "help", None, HELP
    if low in ("", "y", "yes"):
        if vlm_labels is None:
            return "error", None, "블라인드 모드라 VLM 제안이 없습니다: 시작 번호, 'n', 'e ...', 'x' 중 하나를 입력하세요."
        if any(lab is None for lab in vlm_labels):
            return "error", None, "VLM 라벨이 없는 키프레임('?')이 있습니다: 'e ...'나 시작 번호를 입력하세요."
        return "label", list(vlm_labels), ""
    if low == "n":
        return "label", [Label.FAILURE] * n, ""
    if low == "x":
        return "label", [Label.INVALID] * n, ""
    if low.isdigit():
        i = int(low)
        if not 0 <= i < n:
            return "error", None, f"시작 번호는 0부터 {n - 1} 사이여야 합니다."
        return "label", [Label.FAILURE] * i + [Label.SUCCESS] * (n - i), ""
    if low.startswith("e"):
        letters = c[1:].strip().replace(" ", "").upper()
        if len(letters) != n or any(ch not in _LETTER for ch in letters):
            return "error", None, f"S/F/A/I 중에서 정확히 {n}글자가 필요합니다. 입력: {len(letters)}글자 {letters!r}"
        return "label", [_LETTER[ch] for ch in letters], ""
    return "error", None, f"모르는 명령입니다: {cmd!r}. 도움말은 ? 입력"


def render_filmstrip(keyframes: Sequence[Keyframe], vlm_labels: Sequence[Label | None] | None, out_png: Path,
                     camera: str | None = None, cols: int = 8, thumb: int = 160) -> Path | None:
    """Write a grid of keyframe thumbnails (index and, if assisted, the VLM label) with Pillow. None if unavailable."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    rows = (len(keyframes) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, kf in enumerate(keyframes):
        cam = camera or (sorted(kf.frame_refs)[0] if kf.frame_refs else None)
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
        path = kf.frame_refs.get(cam) if cam else None
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb, thumb))
            sheet.paste(img, (x, y + 18))
        except Exception:
            draw.rectangle([x + 2, y + 20, x + thumb - 2, y + thumb + 16], outline=(200, 60, 60))
            draw.text((x + 8, y + 70), "no image", fill=(200, 60, 60))
        tag = f"#{i}"
        if vlm_labels is not None:
            tag += f" VLM:{_SHORT[vlm_labels[i]]}"
        if not kf.auto_valid:
            tag += " AUTO-INVALID"
        draw.text((x + 4, y + 2), tag, fill=(255, 255, 255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def make_display(out_dir: Path, viewer: str = "print", camera: str | None = None,
                 print_fn: Callable[[str], None] = print) -> Callable:
    """Display callback: render the filmstrip, then print its path or open it (xdg-open / open)."""

    def display(episode: Episode, keyframes: Sequence[Keyframe], vlm_labels: Sequence[Label | None] | None) -> None:
        png = render_filmstrip(keyframes, vlm_labels, out_dir / f"{episode.episode_id}.png", camera)
        if png is None:
            print_fn("(Pillow가 없어 키프레임 모음 그림을 만들 수 없습니다. 프레임 파일 경로:)")
            for i, kf in enumerate(keyframes):
                print_fn(f"  #{i}: {dict(kf.frame_refs)}")
            return
        if viewer in ("xdg-open", "open") and shutil.which(viewer):
            subprocess.Popen([viewer, str(png)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print_fn(f"키프레임 모음 그림: {png}")

    return display


class ReviewSession:
    def __init__(
        self,
        episodes: Sequence[Episode],
        keyframes_by_episode: Mapping[str, Sequence[Keyframe]],
        out_path: str | Path,
        reviewer_id: str,
        purpose: str,
        vlm_by_sample: Mapping[str, VLMSuggestion] | None = None,
        order_seed: int = 0,
        data_kind: DataKind = DataKind.REAL,
        session_id: str = "",
        input_fn: Callable[[str], str] | None = None,
        display_fn: Callable | None = None,
        print_fn: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        wallclock: Callable[[], str] = utc_now_iso,
    ):
        if purpose not in ("pool_review", "audit", "audit_repeat"):
            raise ValueError("purpose must be pool_review, audit or audit_repeat")
        if purpose != "pool_review" and vlm_by_sample is not None:
            raise ValueError("blind audit labeling must not receive VLM suggestions")
        if purpose == "pool_review" and vlm_by_sample is None:
            raise ValueError("pool_review is the assisted session: VLM suggestions are required")
        self.episodes = sorted(episodes, key=lambda e: e.episode_id)
        self.kfs = keyframes_by_episode
        self.out_path = Path(out_path)
        self.reviewer_id, self.purpose = reviewer_id, purpose
        self.vlm = vlm_by_sample
        self.order_seed, self.data_kind, self.session_id = order_seed, DataKind(data_kind), session_id
        # resolve builtins.input at call time (not at import time) so it can be redirected
        self.input_fn = input_fn if input_fn is not None else (lambda prompt: input(prompt))
        self.print_fn, self.clock, self.wallclock = print_fn, clock, wallclock
        self.display_fn = display_fn or (lambda ep, kfs, labels: None)

    def done_ids(self) -> set[str]:
        if not self.out_path.exists():
            return set()
        return {r["episode_id"] for r in read_jsonl(self.out_path) if r.get("purpose") == self.purpose}

    def ordered_ids(self) -> list[str]:
        ids = [e.episode_id for e in self.episodes]
        perm = make_rng(self.order_seed, "review_order", self.purpose).permutation(len(ids))
        return [ids[i] for i in perm]

    def run(self, max_episodes: int | None = None) -> list[ReviewRecord]:
        done = self.done_ids()
        pending = [eid for eid in self.ordered_ids() if eid not in done]
        by_id = {e.episode_id: e for e in self.episodes}
        out: list[ReviewRecord] = []
        self.print_fn(f"완료 {len(done)}개, 남음 {len(pending)}개 ({self.purpose}). 도움말은 ? 입력")
        for eid in pending:
            if max_episodes is not None and len(out) >= max_episodes:
                break
            kfs = list(self.kfs[eid])
            vlm_labels = None
            if self.vlm is not None:
                vlm_labels = [None if self.vlm.get(k.sample_id) is None else self.vlm[k.sample_id].label for k in kfs]
            self.display_fn(by_id[eid], kfs, vlm_labels)
            self.print_fn(f"[{len(done) + len(out) + 1}] 에피소드 {eid}: 키프레임 {len(kfs)}개" +
                          ("" if vlm_labels is None else "   VLM: " + "".join(_SHORT[x] for x in vlm_labels)))
            started, t0 = self.wallclock(), self.clock()
            while True:
                action, labels, msg = parse_command(self.input_fn("> "), len(kfs), vlm_labels)
                if action in ("help", "error"):
                    self.print_fn(msg)
                    continue
                break
            if action == "quit":
                break
            if action == "skip":
                continue
            elapsed = self.clock() - t0
            assert labels is not None
            rec = ReviewRecord(
                eid, self.reviewer_id, self.purpose, self.vlm is not None,
                {k.sample_id: (Label.INVALID if not k.auto_valid else lab) for k, lab in zip(kfs, labels)},
                round(elapsed, 3), "measured", self.data_kind, started, self.wallclock(),
                review_order=len(done) + len(out), session_id=self.session_id,
            )
            append_jsonl(self.out_path, rec)
            out.append(rec)
        return out


def stdin_is_interactive() -> bool:
    return sys.stdin.isatty() and os.environ.get("HILREVIEW_NONINTERACTIVE") != "1"
