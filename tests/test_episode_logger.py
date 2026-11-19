import pytest

from hilreview.cli import main
from hilreview.data.io import read_jsonl
from hilreview.data.schema import DataKind
from hilreview.evaluation.episode_logger import EpisodeOutcomeLogger, parse_outcome
from hilreview.evaluation.online import OnlineEpisodeRecord, RunRecord, summarize_run


def test_parse_outcome():
    assert parse_outcome("cy") == ("classifier_success", True, False)
    assert parse_outcome(" CN ") == ("classifier_success", False, False)
    assert parse_outcome("t?") == ("timeout", None, False)
    assert parse_outcome("kyi") == ("human_success_key", True, True)
    for bad in ("c", "cx", "zy", "cyz", "cyii", ""):
        assert isinstance(parse_outcome(bad), str), bad


def _logger(tmp_path, cmds, phase="train", **kw):
    it = iter(cmds)
    printed = []
    lg = EpisodeOutcomeLogger("run-abc", phase, tmp_path / "eps.jsonl", tmp_path / "runs.jsonl", 10.0,
                              DataKind.SYNTHETIC, input_fn=lambda p: next(it), print_fn=printed.append,
                              wallclock=lambda: "2026-01-01T00:00:00Z", **kw)
    return lg, printed


def test_logging_undo_warning_summary_and_run_status(tmp_path):
    lg, printed = _logger(tmp_path, ["cy", "cn", "bad", "tn", "u", "ty", "cyi", "s", "q", "z", "c"])
    rr = lg.run()
    recs = read_jsonl(tmp_path / "eps.jsonl", OnlineEpisodeRecord.from_dict)
    assert [(r.episode_index, r.end_reason, r.human_judged_success, r.human_intervened) for r in recs] == [
        (0, "classifier_success", True, False), (1, "classifier_success", False, False),
        (2, "timeout", True, False), (3, "classifier_success", True, True)]
    assert all(r.logged_at and r.fps == 10.0 and r.data_kind is DataKind.SYNTHETIC for r in recs)
    assert any("가짜 성공 종료로 기록" in p for p in printed)  # live warning for 'cn'
    assert any("놓친 성공" in p for p in printed)  # 'ty'
    assert any("입력 형식이 맞지 않습니다" in p for p in printed)
    assert any("가짜 성공 종료 1" in p for p in printed)  # summary 's'
    assert isinstance(rr, RunRecord) and rr.status == "completed"
    assert read_jsonl(tmp_path / "runs.jsonl", RunRecord.from_dict)[0].run_id == "run-abc"
    s = summarize_run(recs)
    assert s["train_false_success_terminations"]["count"] == 1 and s["train_missed_success_timeouts"] == 1
    assert s["train_interventions_human_reported"]["episodes_with_intervention"] == 1


def test_resume_and_eval_phase_do_not_touch_run_status(tmp_path):
    _logger(tmp_path, ["cy", "q", "a"])[0].run()
    lg, printed = _logger(tmp_path, ["tn", "q", "c"])  # same run, resumed
    lg.run()
    recs = read_jsonl(tmp_path / "eps.jsonl", OnlineEpisodeRecord.from_dict)
    assert [r.episode_index for r in recs] == [0, 1]
    assert any("이어서 기록합니다" in p for p in printed)
    assert read_jsonl(tmp_path / "runs.jsonl", RunRecord.from_dict)[0].status == "completed"  # replaced, not duplicated
    ev, _ = _logger(tmp_path, ["cy", "q"], phase="eval")
    assert ev.run() is None  # eval sessions never overwrite the training run's status
    recs = read_jsonl(tmp_path / "eps.jsonl", OnlineEpisodeRecord.from_dict)
    assert [(r.phase, r.episode_index) for r in recs][-1] == ("eval", 0)
    assert len(read_jsonl(tmp_path / "runs.jsonl")) == 1


def test_cli_log_episodes(tmp_path, monkeypatch, capsys):
    answers = iter(["cy", "cn", "q", "c"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["log-episodes", "--run-id", "run-x", "--phase", "train", "--out", str(tmp_path / "e.jsonl"),
                 "--runs-out", str(tmp_path / "r.jsonl"), "--fps", "10", "--data-kind", "synthetic"]) == 0
    out = capsys.readouterr().out
    assert "연습 모드" in out and "실행 상태: completed" in out
    assert len(read_jsonl(tmp_path / "e.jsonl")) == 2
