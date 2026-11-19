"""End-to-end protocol tests on small SYNTHETIC experiments."""

import copy
import json

import pytest

from hilreview.data.schema import Label, ReviewRecord
from hilreview.experiments.config import ConfigError, validate_config
from hilreview.experiments.runner import prepare_data, run_conditions, run_experiment
from hilreview.labeling.aggregate import ConditionSpec, build_condition_labels
from hilreview.utils.numeric import round_half_up

import numpy as np

BASE = {
    "experiment_name": "test_e2e",
    "data": {"kind": "synthetic", "synthetic": {"n_episodes": 48, "n_sessions": 3, "seed": 3}},
    "conditions": [
        {"name": "A", "policy": "full", "fraction": 1.0},
        {"name": "B", "policy": "random", "fraction": 0.25},
        {"name": "C", "policy": "none", "fraction": 0.0},
        {"name": "B_low", "policy": "low_confidence", "fraction": 0.25},
        {"name": "B_fake", "policy": "fake_success_priority", "fraction": 0.25},
    ],
    "replicates": {"n": 2, "base_seed": 11},
    "bootstrap": {"n_boot": 100, "seed": 1},
}


@pytest.fixture(scope="module")
def cfg():
    return validate_config(copy.deepcopy(BASE))


@pytest.fixture(scope="module")
def data(cfg):
    return prepare_data(cfg)


@pytest.fixture(scope="module")
def rows(data, cfg):
    return run_conditions(data, cfg)


def _pool_samples(data):
    return [kf for eid in data.ids("train", "val") for kf in data.keyframes_by_episode[eid] if kf.auto_valid]


def test_conditions_follow_protocol(data, cfg):
    pool = data.ids("train", "val")
    n = len(pool)
    assert n > 0
    k25 = round_half_up(0.25 * n)
    for rep_seed in (0, 1):
        for spec_d in cfg["conditions"]:
            spec = ConditionSpec(spec_d["name"], spec_d["policy"], spec_d["fraction"])
            cl = build_condition_labels(spec, pool, data.keyframes_by_episode, data.vlm_by_sample,
                                        data.pool_reviews, np.random.default_rng(rep_seed))
            sel = set(cl.selected_episodes)
            assert sel <= set(pool), "only pool episodes may be reviewed"
            assert not (sel & set(data.ids("audit"))), "audit episodes are never reviewed for training"
            expected_k = {"full": n, "none": 0}.get(spec.policy, k25)
            assert cl.k == expected_k and len(sel) == expected_k, spec.name
            for kf in _pool_samples(data):
                src = cl.sources[kf.sample_id]
                if kf.episode_id in sel:
                    assert src == "human"
                    assert cl.labels[kf.sample_id] is data.pool_reviews[kf.episode_id].labels[kf.sample_id]
                else:
                    v = data.vlm_by_sample.get(kf.sample_id)
                    if v is None or v.label is None:
                        assert src == "none" and cl.labels[kf.sample_id] is None
                    else:
                        assert src == "vlm" and cl.labels[kf.sample_id] is v.label
            if spec.policy == "none":
                assert all(s in ("vlm", "none", "auto_invalid") for s in cl.sources.values())
            if spec.policy == "full":
                assert all(s in ("human", "auto_invalid") for s in cl.sources.values())


def test_rows_report_exact_budget(rows, data):
    n = len(data.ids("train", "val"))
    for r in rows:
        assert r["budget"]["n_eligible_pool_episodes"] == n
        assert len(r["selected_episodes"]) == r["budget"]["k"]
        if r["policy"] == "none":
            assert r["effort"]["human_time_s"] == 0.0 and r["effort"]["timing_source"] == "not_applicable"
        else:
            assert r["effort"]["timing_source"] == "simulated"


def test_random_selection_varies_by_replicate_but_deterministic_policies_do_not(rows, data):
    from hilreview.labeling.aggregate import episode_views
    from hilreview.review.policies import episode_uncertainty

    sel = {(r["condition"], r["replicate"]): tuple(r["selected_episodes"]) for r in rows}
    assert sel[("B", 0)] != sel[("B", 1)]
    assert sel[("A", 0)] == sel[("A", 1)]
    # low_confidence must equal the top-k by episode uncertainty (unless a tie straddles the cut-off)
    views = episode_views(data.ids("train", "val"), data.keyframes_by_episode, data.vlm_by_sample)
    scores = sorted(((episode_uncertainty(v), v.episode_id) for v in views), reverse=True)
    k = len(sel[("B_low", 0)])
    if scores[k - 1][0] != scores[k][0]:
        expected = {eid for _, eid in scores[:k]}
        assert set(sel[("B_low", 0)]) == expected == set(sel[("B_low", 1)])


def test_audit_labels_cannot_influence_anything_upstream(data, cfg, rows):
    """Flip every audit label: selections, labels, models and thresholds must be bit-identical."""
    flipped = copy.copy(data)
    flip = {Label.SUCCESS: Label.FAILURE, Label.FAILURE: Label.SUCCESS}
    flipped.audit_reviews = {
        eid: ReviewRecord(**{**rec.__dict__, "labels": {s: flip.get(l, l) for s, l in rec.labels.items()}})
        for eid, rec in data.audit_reviews.items()
    }
    rows2 = run_conditions(flipped, cfg)
    assert len(rows2) == len(rows)
    changed_audit = 0
    for a, b in zip(rows, rows2):
        assert a["fingerprints"] == b["fingerprints"], a["condition"]
        assert a["threshold"] == b["threshold"]
        assert a["label_counts"] == b["label_counts"]
        assert a["effort"] == b["effort"]
        changed_audit += a["audit"]["keyframe"]["accuracy"] != b["audit"]["keyframe"]["accuracy"]
    assert changed_audit == len(rows), "audit metrics must depend on audit labels (sanity check of the test)"


def test_selection_never_reads_human_labels(data):
    """Corrupt all pool review labels: which episodes get selected must not change."""
    corrupted = {eid: ReviewRecord(**{**rec.__dict__, "labels": {s: Label.AMBIGUOUS for s in rec.labels}})
                 for eid, rec in data.pool_reviews.items()}
    pool = data.ids("train", "val")
    for policy in ("random", "low_confidence", "fake_success_priority"):
        spec = ConditionSpec("x", policy, 0.25)
        a = build_condition_labels(spec, pool, data.keyframes_by_episode, data.vlm_by_sample, data.pool_reviews,
                                   np.random.default_rng(5))
        b = build_condition_labels(spec, pool, data.keyframes_by_episode, data.vlm_by_sample, corrupted,
                                   np.random.default_rng(5))
        assert a.selected_episodes == b.selected_episodes


def test_full_run_is_deterministic(cfg, tmp_path):
    r1 = run_experiment(cfg, tmp_path / "r1")
    r2 = run_experiment(cfg, tmp_path / "r2")
    assert json.dumps(r1["rows"], sort_keys=True) == json.dumps(r2["rows"], sort_keys=True)
    assert (tmp_path / "r1" / "per_replicate.json").read_text() == (tmp_path / "r2" / "per_replicate.json").read_text()
    assert (tmp_path / "r1" / "summary.json").read_text() == (tmp_path / "r2" / "summary.json").read_text()
    report = (tmp_path / "r1" / "report.md").read_text()
    assert "경고: 합성(가짜) 데이터" in report and "없음(MISSING)" in report  # default report_lang = ko
    prov = json.loads((tmp_path / "r1" / "provenance.json").read_text())
    assert prov["data_kind"] == "synthetic" and prov["config_hash"]


def _with(**over):
    c = copy.deepcopy(BASE)
    for path, value in over.items():
        node = c
        keys = path.split("__")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    c["conditions"] = [x for x in c["conditions"] if x["name"] in ("A", "C")]
    return validate_config(c)


def test_positive_control_noisy_vlm_hurts_vlm_only_condition():
    cfg = _with(vlm__mock={"base_error": 0.35, "hover_fake_success": 1.0, "boundary_error": 0.5},
                data__synthetic={"n_episodes": 60, "n_sessions": 3, "seed": 4})
    rows = run_conditions(prepare_data(cfg), cfg)
    a = [r for r in rows if r["condition"] == "A"]
    c = [r for r in rows if r["condition"] == "C"]
    for ra, rc in zip(a, c):
        assert rc["label_quality_vs_synthetic_truth"]["fpr"] > ra["label_quality_vs_synthetic_truth"]["fpr"] + 0.15
        assert rc["audit"]["keyframe"]["fpr"] > ra["audit"]["keyframe"]["fpr"]


def test_negative_control_perfect_vlm_matches_truth():
    cfg = _with(vlm__mock={"base_error": 0.0, "boundary_error": 0.0, "occluded_error": 0.0,
                           "hover_fake_success": 0.0, "unsure_rate": 0.0, "request_error_rate": 0.0})
    rows = run_conditions(prepare_data(cfg), cfg)
    for r in rows:
        if r["condition"] == "C":
            assert r["label_quality_vs_synthetic_truth"]["accuracy"] == 1.0
            assert r["label_counts"]["no_label"] == 0


def test_config_validation_rejects_inconsistent_conditions():
    bad = copy.deepcopy(BASE)
    bad["conditions"] = [{"name": "A", "policy": "full", "fraction": 0.5}]
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad["conditions"] = [{"name": "B", "policy": "random", "fraction": 1.0}]
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad = copy.deepcopy(BASE)
    bad["data"] = {"kind": "real", "real": {}}
    with pytest.raises(ConfigError):
        validate_config(bad)
