"""Tests for issues found in the code-review pass (leakage guards, statistics, data-safety guards)."""

import copy

import numpy as np
import pytest

from hilreview.data.schema import DataKind, Keyframe, ReviewRecord, VLMSuggestion, make_sample_id
from hilreview.evaluation.metrics import BinaryCounts, paired_cluster_bootstrap_diff
from hilreview.experiments.config import validate_config
from hilreview.experiments.report import render_report
from hilreview.experiments.runner import (
    check_review_placement,
    global_audit_metrics,
    prepare_data,
    run_conditions,
    run_experiment,
)
from hilreview.labeling.vlm import OpenAICompatConfig, OpenAICompatVLMLabeler, is_local_or_private_url

CFG = {
    "experiment_name": "review_fixes",
    "data": {"kind": "synthetic", "synthetic": {"n_episodes": 40, "n_sessions": 2, "seed": 9,
                                                 "auto_invalid_keyframe_prob": 0.05}},
    "conditions": [{"name": "A", "policy": "full", "fraction": 1.0},
                   {"name": "B", "policy": "low_confidence", "fraction": 0.25},
                   {"name": "C", "policy": "none", "fraction": 0.0}],
    "replicates": {"n": 1, "base_seed": 2},
    "bootstrap": {"n_boot": 200, "seed": 0},
}


@pytest.fixture(scope="module")
def cfg():
    return validate_config(copy.deepcopy(CFG))


@pytest.fixture(scope="module")
def data(cfg):
    return prepare_data(cfg)


def test_paired_bootstrap_identical_and_shifted():
    a = [BinaryCounts(tp=1, tn=3)] * 10 + [BinaryCounts(fp=1, tn=2, tp=1)] * 10
    same = paired_cluster_bootstrap_diff(a, a, "fpr", n_boot=300, seed=1)
    assert same["diff"] == 0.0 and same["lo"] == 0.0 and same["hi"] == 0.0
    worse = [BinaryCounts(fp=c.fp + 1, tn=max(c.tn - 1, 0), tp=c.tp) for c in a]
    d = paired_cluster_bootstrap_diff(a, worse, "fpr", n_boot=300, seed=1)
    assert d["diff"] > 0 and d["lo"] > 0  # every episode got worse -> interval excludes 0
    with pytest.raises(ValueError):
        paired_cluster_bootstrap_diff(a, a[:-1], "fpr")


def test_paired_rows_and_report_sections(cfg, tmp_path):
    res = run_experiment(cfg, tmp_path / "out")
    rows = {r["condition"]: r for r in res["rows"]}
    assert "paired_vs_reference" not in rows["A"]
    pv = rows["C"]["paired_vs_reference"]
    assert pv["reference"] == "A" and set(pv) >= {"accuracy", "fpr", "fnr", "fst_rate"}
    assert "_per_episode" not in rows["C"]["audit"]  # private data is not written out
    report = (tmp_path / "out" / "report.md").read_text()
    assert "`A` 대비 짝지은 차이" in report and "불확실성 읽는 법" in report
    assert "자른 사람 시간" in report


def test_agreement_excludes_auto_invalid_keyframes(data):
    g = global_audit_metrics(data)["audit_repeat_agreement"]
    expected = sum(1 for eid in data.audit_repeat for kf in data.keyframes_by_episode[eid] if kf.auto_valid)
    total = sum(len(data.keyframes_by_episode[eid]) for eid in data.audit_repeat)
    assert g["n"] == expected and expected < total  # the fixture does contain auto-invalid keyframes


def test_confidence_policies_refuse_missing_scores(data, cfg):
    no_p = copy.copy(data)
    no_p.vlm_by_sample = {k: VLMSuggestion(**{**v.__dict__, "p_success": None, "confidence_source": "none"})
                          for k, v in data.vlm_by_sample.items()}
    with pytest.raises(ValueError, match="no p_success"):
        run_conditions(no_p, cfg)
    cfg_rand = validate_config({**copy.deepcopy(CFG), "conditions": [{"name": "R", "policy": "random",
                                                                       "fraction": 0.25}]})
    run_conditions(no_p, cfg_rand)  # random selection does not need scores


def test_review_placement_checks(data):
    check_review_placement(data)  # the simulated data satisfies the protocol
    audit_eid = data.ids("audit")[0]
    leaked = copy.copy(data)
    rec = data.audit_reviews[audit_eid]
    leaked.pool_reviews = {**data.pool_reviews,
                           audit_eid: ReviewRecord(**{**rec.__dict__, "purpose": "pool_review", "shown_vlm": True})}
    with pytest.raises(ValueError, match="WITH VLM suggestions"):
        check_review_placement(leaked)
    missing = copy.copy(data)
    missing.pool_reviews = dict(list(data.pool_reviews.items())[1:])
    with pytest.raises(ValueError, match="lack a pool_review"):
        check_review_placement(missing)
    partial = copy.copy(data)
    eid = data.ids("train")[0]
    r = data.pool_reviews[eid]
    partial.pool_reviews = {**data.pool_reviews,
                            eid: ReviewRecord(**{**r.__dict__, "labels": dict(list(r.labels.items())[:1])})}
    with pytest.raises(ValueError, match="does not cover"):
        check_review_placement(partial)


@pytest.mark.parametrize("url,ok", [
    ("http://127.0.0.1:8000/v1", True), ("http://localhost:11434/v1", True), ("http://192.168.1.20:8000/v1", True),
    ("http://10.0.0.5/v1", True), ("http://gpu-laptop.local:8000/v1", True), ("http://[::1]:8000/v1", True),
    ("https://api.example.com/v1", False), ("http://8.8.8.8/v1", False), ("http://my-vlm-host:8000/v1", False),
])
def test_local_url_guard(url, ok):
    assert is_local_or_private_url(url) is ok


def test_labeler_refuses_remote_host_unless_allowed():
    with pytest.raises(ValueError, match="refusing"):
        OpenAICompatVLMLabeler(OpenAICompatConfig(base_url="https://api.example.com/v1"), task="t")
    OpenAICompatVLMLabeler(OpenAICompatConfig(base_url="https://api.example.com/v1"), task="t",
                           allow_remote_host=True)


def test_report_languages_differ_only_in_wording(cfg, tmp_path):
    """ko and en reports must contain exactly the same numbers in the same order; only the words change."""
    import re

    from hilreview.experiments.config import ConfigError

    res = run_experiment(cfg, None)
    ko = render_report(res["provenance"], res["summary"], {**cfg, "report_lang": "ko"})
    en = render_report(res["provenance"], res["summary"], {**cfg, "report_lang": "en"})
    nums = lambda text: re.findall(r"-?\d+\.\d{3}", text)  # noqa: E731
    assert nums(ko) == nums(en) and len(nums(ko)) > 50
    assert "경고: 합성(가짜) 데이터" in ko and "WARNING: SYNTHETIC DATA" in en
    assert "Paired differences vs `A`" in en and "How to read the uncertainty" in en
    with pytest.raises(ValueError):
        render_report(res["provenance"], res["summary"], {**cfg, "report_lang": "fr"})
    with pytest.raises(ConfigError):
        validate_config({**copy.deepcopy(CFG), "report_lang": "jp"})
