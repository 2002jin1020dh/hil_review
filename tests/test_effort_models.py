import numpy as np
import pytest

from hilreview.data.schema import DataKind, Keyframe, Label, ReviewRecord, VLMSuggestion, make_sample_id
from hilreview.evaluation.effort import summarize_review_effort, summarize_vlm_cost
from hilreview.models.logistic import LogisticRegressionSGD
from hilreview.models.threshold import select_threshold

S, F = Label.SUCCESS, Label.FAILURE


def _setup():
    kfs = {f"e{i}": [Keyframe(make_sample_id(f"e{i}", j), f"e{i}", j, j / 10.0) for j in range(4)] for i in range(3)}
    vlm = {}
    for eid, ks in kfs.items():
        for kf in ks:
            vlm[kf.sample_id] = VLMSuggestion(kf.sample_id, eid, F, 0.2, "mock", "m", "v", "ok", DataKind.SYNTHETIC)
    reviews = {
        "e0": ReviewRecord("e0", "r", "pool_review", True, {k.sample_id: F for k in kfs["e0"]}, 10.0, "measured",
                           DataKind.REAL),
        "e1": ReviewRecord("e1", "r", "pool_review", True,
                           {k.sample_id: (S if k.frame_index >= 2 else F) for k in kfs["e1"]}, 500.0, "measured",
                           DataKind.REAL),
        "e2": ReviewRecord("e2", "r", "pool_review", True, {k.sample_id: F for k in kfs["e2"]}, None, "missing",
                           DataKind.REAL),
    }
    return kfs, vlm, reviews


def test_effort_counts_corrections_caps_idle_and_reports_missing():
    kfs, vlm, reviews = _setup()
    e = summarize_review_effort(["e0", "e1"], reviews, kfs, vlm, idle_cap_s=120.0)
    assert e["n_episodes_reviewed"] == 2 and e["n_keyframes_reviewed"] == 8
    assert e["n_corrections"] == 2 and e["correction_rate"] == pytest.approx(0.25)
    assert e["human_time_s"] == 510.0 and e["human_time_s_capped"] == 130.0 and e["n_time_capped"] == 1
    assert e["timing_source"] == "measured" and e["s_per_keyframe"] == pytest.approx(510 / 8)
    m = summarize_review_effort(["e0", "e2"], reviews, kfs, vlm)
    assert m["n_time_missing"] == 1 and m["timing_source"] == "mixed" and m["s_per_keyframe"] is None
    z = summarize_review_effort([], reviews, kfs, vlm)
    assert z["human_time_s"] == 0.0 and z["timing_source"] == "not_applicable"


def test_vlm_cost():
    kfs, vlm, _ = _setup()
    sids = [k.sample_id for ks in kfs.values() for k in ks]
    c = summarize_vlm_cost(sids, vlm)
    assert c["n_calls"] == 12 and c["n_errors"] == 0 and c["total_latency_s"] is None
    assert c["latency_source"] == "missing"


def test_threshold_fixed_and_fbeta():
    assert select_threshold({"policy": "fixed", "value": 0.7}, [], [])[0] == 0.7
    scores = [0.1, 0.2, 0.35, 0.45, 0.62, 0.8, 0.9]
    targets = [0, 0, 0, 1, 0, 1, 1]
    thr, info = select_threshold({"policy": "val_fbeta", "beta": 0.5}, scores, targets)
    # >0.62 and <0.8 gives precision 1, recall 2/3: F0.5 = 0.909, the best on the grid; ties -> largest threshold
    assert thr == pytest.approx(0.75) and info["val_fbeta"] == pytest.approx((1.25 * 1 * 2 / 3) / (0.25 + 2 / 3))
    thr, info = select_threshold({"policy": "val_fbeta"}, [0.3, 0.4], [0, 0])
    assert info["fallback"] and thr == 0.5
    with pytest.raises(ValueError):
        select_threshold({"policy": "audit_tuned"}, [], [])


def test_logistic_learns_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    m1 = LogisticRegressionSGD(seed=3).fit(X, y)
    m2 = LogisticRegressionSGD(seed=3).fit(X, y)
    acc = np.mean((m1.predict_proba(X) > 0.5) == y)
    assert acc > 0.9 and m1.param_hash() == m2.param_hash()
    assert LogisticRegressionSGD(seed=4).fit(X, y).param_hash() != m1.param_hash()
    deg = LogisticRegressionSGD().fit(X, np.zeros(400))
    assert deg.degenerate and np.all(deg.predict_proba(X) < 0.01)
    with pytest.raises(ValueError):
        LogisticRegressionSGD().fit(X, np.full(400, 2))
