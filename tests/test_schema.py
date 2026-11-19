import json

import pytest

from hilreview.data.io import append_jsonl, read_jsonl, write_jsonl
from hilreview.data.schema import (
    DataKind,
    Episode,
    Keyframe,
    Label,
    ReviewRecord,
    VLMSuggestion,
    label_to_target,
    make_sample_id,
)


def _ep(**kw):
    d = dict(episode_id="e1", session_id="s1", collection_mode="success_attempt", n_frames=100, fps=10.0,
             data_kind=DataKind.SYNTHETIC)
    d.update(kw)
    return Episode(**d)


def test_label_parse_and_targets():
    assert Label.parse(" Success ") is Label.SUCCESS
    assert Label.parse(Label.FAILURE) is Label.FAILURE
    with pytest.raises(ValueError):
        Label.parse("maybe")
    assert label_to_target(Label.SUCCESS) == 1
    assert label_to_target(Label.FAILURE) == 0
    assert label_to_target(Label.AMBIGUOUS) is None
    assert label_to_target(Label.INVALID) is None
    assert label_to_target(None) is None


def test_roundtrip_all_records(tmp_path):
    ep = _ep()
    kf = Keyframe(make_sample_id("e1", 5), "e1", 5, 0.5, {"front": "a.png"}, (0.1, 0.2), True)
    sug = VLMSuggestion(kf.sample_id, "e1", Label.SUCCESS, 0.8, "logprob", "m", "v1", "ok", DataKind.REAL, 1.2,
                        "measured", "yes")
    bad = VLMSuggestion(kf.sample_id, "e1", None, None, "none", "m", "v1", "parse_error", DataKind.REAL)
    rev = ReviewRecord("e1", "r1", "pool_review", True, {kf.sample_id: Label.FAILURE}, 3.5, "measured",
                       DataKind.REAL, review_order=0)
    for obj, cls in [(ep, Episode), (kf, Keyframe), (sug, VLMSuggestion), (bad, VLMSuggestion), (rev, ReviewRecord)]:
        path = tmp_path / f"{cls.__name__}.jsonl"
        write_jsonl(path, [obj])
        (back,) = read_jsonl(path, cls.from_dict)
        assert back == obj
        assert json.loads(path.read_text())  # plain JSON
    append_jsonl(tmp_path / "Episode.jsonl", ep)
    assert len(read_jsonl(tmp_path / "Episode.jsonl")) == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _ep(n_frames=0),
        lambda: _ep(eligible=False),  # no reason given
        lambda: Keyframe("e1/f000001", "e1", 2, 0.2),  # id/frame mismatch
        lambda: VLMSuggestion("e1/f000001", "e1", Label.SUCCESS, 0.5, "logprob", "m", "v", "parse_error",
                              DataKind.REAL),  # failed but labeled
        lambda: VLMSuggestion("e1/f000001", "e1", Label.SUCCESS, 1.5, "logprob", "m", "v", "ok", DataKind.REAL),
        lambda: ReviewRecord("e1", "r", "audit", True, {"e1/f000001": Label.SUCCESS}, 1.0, "measured",
                             DataKind.REAL),  # audit must be blind
        lambda: ReviewRecord("e1", "r", "pool_review", True, {"e1/f000001": Label.SUCCESS}, 1.0, "missing",
                             DataKind.REAL),  # missing timing with a value
        lambda: ReviewRecord("e1", "r", "pool_review", True, {}, None, "missing", DataKind.REAL),  # no labels
    ],
)
def test_validation_rejects_bad_records(factory):
    with pytest.raises(ValueError):
        factory()
