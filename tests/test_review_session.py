import pytest

from hilreview.data.io import read_jsonl
from hilreview.data.schema import DataKind, Episode, Keyframe, Label, ReviewRecord, VLMSuggestion, make_sample_id
from hilreview.review.session import ReviewSession, parse_command, render_filmstrip

S, F, A, I = Label.SUCCESS, Label.FAILURE, Label.AMBIGUOUS, Label.INVALID


def test_parse_command_variants():
    vlm = [F, F, S, S]
    assert parse_command("", 4, vlm) == ("label", vlm, "")
    assert parse_command("y", 4, vlm)[1] == vlm
    assert parse_command("2", 4, None)[1] == [F, F, S, S]
    assert parse_command("0", 4, None)[1] == [S, S, S, S]
    assert parse_command("n", 4, None)[1] == [F] * 4
    assert parse_command("x", 4, None)[1] == [I] * 4
    assert parse_command("e ffas", 4, None)[1] == [F, F, A, S]
    assert parse_command("q", 4, None)[0] == "quit" and parse_command("s", 4, None)[0] == "skip"
    for bad in ("4", "e FFF", "e FFFQ", "maybe"):
        assert parse_command(bad, 4, vlm)[0] == "error", bad
    assert parse_command("", 4, None)[0] == "error"  # blind mode cannot "accept"
    assert parse_command("y", 4, [F, None, S, S])[0] == "error"


def _data(n_eps=3, n_kf=4):
    eps = [Episode(f"e{i}", "s", "success_attempt", 40, 10.0, DataKind.REAL) for i in range(n_eps)]
    kfs = {e.episode_id: [Keyframe(make_sample_id(e.episode_id, j * 10), e.episode_id, j * 10, float(j),
                                   auto_valid=not (e.episode_id == "e0" and j == 3)) for j in range(n_kf)]
           for e in eps}
    vlm = {k.sample_id: VLMSuggestion(k.sample_id, k.episode_id, F, 0.1, "logprob", "m", "v", "ok", DataKind.REAL)
           for ks in kfs.values() for k in ks}
    return eps, kfs, vlm


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 2.5
        return self.t


def test_assisted_session_records_labels_time_and_resumes(tmp_path):
    eps, kfs, vlm = _data()
    out = tmp_path / "pool_reviews.jsonl"
    cmds = iter(["?", "bogus", "2", "q"])
    printed = []
    sess = ReviewSession(eps, kfs, out, "alice", "pool_review", vlm, order_seed=1, input_fn=lambda p: next(cmds),
                         print_fn=printed.append, clock=FakeClock(), wallclock=lambda: "2026-01-01T00:00:00Z")
    recs = sess.run()
    assert len(recs) == 1
    r = recs[0]
    first = sess.ordered_ids()[0]
    assert r.episode_id == first and r.timing_source == "measured" and r.shown_vlm
    assert r.active_time_s == pytest.approx(2.5)  # one clock read before, one after the valid command
    expected = [F, F, S, S]
    for k, lab in zip(kfs[first], expected):
        assert r.labels[k.sample_id] is (I if not k.auto_valid else lab)
    assert any("명령어" in p for p in printed) and any("모르는 명령" in p for p in printed)
    # resume: the finished episode is not shown again
    cmds2 = iter(["", "n"])
    sess2 = ReviewSession(eps, kfs, out, "alice", "pool_review", vlm, order_seed=1, input_fn=lambda p: next(cmds2),
                          print_fn=lambda s: None, clock=FakeClock())
    recs2 = sess2.run()
    assert [x.episode_id for x in recs2] == sess.ordered_ids()[1:]
    saved = read_jsonl(out, ReviewRecord.from_dict)
    assert [x.review_order for x in saved] == [0, 1, 2]
    assert len({x.episode_id for x in saved}) == 3


def test_blind_session_rules(tmp_path):
    eps, kfs, vlm = _data()
    with pytest.raises(ValueError):
        ReviewSession(eps, kfs, tmp_path / "a.jsonl", "bob", "audit", vlm_by_sample=vlm)
    with pytest.raises(ValueError):
        ReviewSession(eps, kfs, tmp_path / "a.jsonl", "bob", "pool_review", vlm_by_sample=None)
    cmds = iter(["", "1", "s", "q"])
    sess = ReviewSession(eps, kfs, tmp_path / "a.jsonl", "bob", "audit", input_fn=lambda p: next(cmds),
                         print_fn=lambda s: None, clock=FakeClock())
    recs = sess.run()
    assert len(recs) == 1 and not recs[0].shown_vlm and recs[0].purpose == "audit"


def test_filmstrip_renders_with_missing_images(tmp_path):
    pytest.importorskip("PIL")
    eps, kfs, vlm = _data(n_eps=1, n_kf=10)
    png = render_filmstrip(kfs["e0"], [F] * 10, tmp_path / "f.png")
    assert png is not None and png.exists() and png.stat().st_size > 0
