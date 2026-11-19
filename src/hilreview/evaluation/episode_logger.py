"""Episode-outcome logger for REAL robot runs (online HIL-SERL training and evaluation).

The human observer types two letters (+ Enter) after every episode:
  1st letter, how the episode ended:   c = classifier ended it (success signal)   t = time limit
                                       k = human pressed the success key          f = human pressed the failure key
                                       a = aborted
  2nd letter, did the task REALLY succeed (human judgment, LABEL_DEFINITION.md):  y = yes   n = no   ? = cannot tell
  optional 3rd letter i = a human intervention happened in this episode
Examples: "cy" correct classifier stop, "cn" FALSE-SUCCESS termination, "tn" timeout failure, "cyi" ... with intervention.
Other commands: u = undo last, s = summary so far, ? = help, q = finish the run (asks for the run status).

The logger shows only the opaque run_id (blinding, DECISIONS.md D-13). It never talks to the robot or LeRobot.
Records are rewritten atomically after every change, so a crash loses at most the line being typed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from hilreview.data.io import read_jsonl, write_jsonl
from hilreview.data.schema import DataKind
from hilreview.evaluation.online import OnlineEpisodeRecord, RunRecord, summarize_run
from hilreview.utils.provenance import utc_now_iso

END_KEYS = {"c": "classifier_success", "t": "timeout", "k": "human_success_key", "f": "human_failure_key",
            "a": "abort"}
JUDGE_KEYS = {"y": True, "n": False, "?": None}
STATUS_KEYS = {"c": "completed", "a": "aborted", "x": "crashed"}

HELP_KO = """에피소드가 끝날 때마다 글자 2개(+ 선택 1개)를 입력하고 Enter:
  첫 글자, 어떻게 끝났나:  c 분류기가 끝냄(성공 신호)   t 시간 초과   k 사람이 성공 키   f 사람이 실패 키   a 중단
  둘째 글자, 실제로 성공했나 (LABEL_DEFINITION.md 기준):  y 예   n 아니오   ? 판단 불가
  셋째 글자(선택): i = 이 에피소드에서 사람이 개입했음
  예) cy = 분류기가 맞게 끝냄,  cn = 가짜 성공 종료!,  tn = 시간 초과 실패,  cyi = 개입 있었음
기타:  u 마지막 기록 취소   s 지금까지 요약   ? 도움말   q 이 실행 끝내기 (실행 상태를 물어봄)
"""


def parse_outcome(cmd: str) -> tuple[str, bool | None, bool] | str:
    """Return (end_reason, human_judged_success, intervened) or an error message (str)."""
    c = cmd.strip().lower()
    if len(c) not in (2, 3) or c[0] not in END_KEYS or c[1] not in JUDGE_KEYS or (len(c) == 3 and c[2] != "i"):
        return f"입력 형식이 맞지 않습니다: {cmd!r}  (예: cy, cn, tn, cyi). 도움말은 ?"
    return END_KEYS[c[0]], JUDGE_KEYS[c[1]], len(c) == 3


class EpisodeOutcomeLogger:
    def __init__(
        self,
        run_id: str,
        phase: str,
        out_path: str | Path,
        runs_path: str | Path | None = None,
        fps: float | None = None,
        data_kind: DataKind = DataKind.REAL,
        input_fn: Callable[[str], str] | None = None,
        print_fn: Callable[[str], None] = print,
        wallclock: Callable[[], str] = utc_now_iso,
        clock: Callable[[], float] = time.monotonic,
    ):
        if phase not in ("train", "eval"):
            raise ValueError("phase must be 'train' or 'eval'")
        self.run_id, self.phase, self.fps = run_id, phase, fps
        self.out_path = Path(out_path)
        self.runs_path = Path(runs_path) if runs_path else None
        self.data_kind = DataKind(data_kind)
        self.input_fn = input_fn if input_fn is not None else (lambda prompt: input(prompt))
        self.print_fn, self.wallclock, self.clock = print_fn, wallclock, clock
        self.all_records: list[OnlineEpisodeRecord] = (
            read_jsonl(self.out_path, OnlineEpisodeRecord.from_dict) if self.out_path.exists() else []
        )

    def _mine(self) -> list[OnlineEpisodeRecord]:
        return [r for r in self.all_records if r.run_id == self.run_id and r.phase == self.phase]

    def _save(self) -> None:
        write_jsonl(self.out_path, self.all_records)

    def _summary_text(self) -> str:
        mine = self._mine()
        if not mine:
            return "아직 기록 없음"
        s = summarize_run([r for r in self.all_records if r.run_id == self.run_id])
        fst = s[f"{self.phase}_false_success_terminations"]["count"]
        clf = sum(1 for r in mine if r.end_reason == "classifier_success")
        succ = sum(1 for r in mine if r.human_judged_success)
        return (f"[{self.run_id}/{self.phase}] 에피소드 {len(mine)}개 | 실제 성공 {succ} | 분류기 종료 {clf} | "
                f"가짜 성공 종료 {fst} | 개입 보고 {sum(1 for r in mine if r.human_intervened)}")

    def run(self) -> RunRecord | None:
        t0, started = self.clock(), self.wallclock()
        self.print_fn(HELP_KO)
        if self._mine():
            self.print_fn(f"이어서 기록합니다. {self._summary_text()}")
        while True:
            idx = len(self._mine())
            cmd = self.input_fn(f"[{self.run_id}/{self.phase}] 에피소드 {idx} 결과> ").strip()
            low = cmd.lower()
            if low in ("?", "h", "help"):
                self.print_fn(HELP_KO)
                continue
            if low == "s":
                self.print_fn(self._summary_text())
                continue
            if low == "u":
                mine = self._mine()
                if not mine:
                    self.print_fn("취소할 기록이 없습니다.")
                    continue
                self.all_records.remove(mine[-1])
                self._save()
                self.print_fn(f"에피소드 {mine[-1].episode_index} 기록을 취소했습니다.")
                continue
            if low == "q":
                return self._finish(t0, started)
            parsed = parse_outcome(cmd)
            if isinstance(parsed, str):
                self.print_fn(parsed)
                continue
            end_reason, judged, intervened = parsed
            rec = OnlineEpisodeRecord(self.run_id, self.phase, idx, end_reason, judged, self.data_kind,
                                      fps=self.fps, logged_at=self.wallclock(), human_intervened=intervened)
            self.all_records.append(rec)
            self._save()
            if end_reason == "classifier_success" and judged is False:
                self.print_fn("⚠ 가짜 성공 종료로 기록했습니다 (분류기는 성공이라 했지만 실제는 실패).")
            if end_reason == "timeout" and judged is True:
                self.print_fn("참고: 놓친 성공으로 기록했습니다 (실제 성공했지만 분류기가 끝내지 않음).")

    def _finish(self, t0: float, started: str) -> RunRecord | None:
        self.print_fn(self._summary_text())
        if self.runs_path is None or self.phase != "train":  # the run status belongs to the training run
            return None
        while True:
            k = self.input_fn("이 실행의 상태는? c 정상 완료 / a 중간에 멈춤 / x 프로그램 오류로 죽음 > ").strip().lower()
            if k in STATUS_KEYS:
                break
            self.print_fn("c, a, x 중 하나를 입력하세요.")
        rr = RunRecord(self.run_id, STATUS_KEYS[k], self.data_kind,
                       wall_clock_s=round(self.clock() - t0, 1),
                       notes=f"logger session {started} -> {self.wallclock()} (phase {self.phase}); "
                             "wall_clock_s = logging-session length, not the exact RL run length")
        runs = read_jsonl(self.runs_path, RunRecord.from_dict) if self.runs_path.exists() else []
        runs = [r for r in runs if r.run_id != self.run_id] + [rr]
        write_jsonl(self.runs_path, runs)
        return rr
