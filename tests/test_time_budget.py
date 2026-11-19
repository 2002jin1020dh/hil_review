"""Equal-TIME review budgets: review in priority order until the time budget is used up."""

import copy

import numpy as np
import pytest

from hilreview.data.schema import Label, ReviewRecord
from hilreview.experiments.config import ConfigError, validate_config
from hilreview.experiments.runner import prepare_data, run_conditions
from hilreview.labeling.aggregate import ConditionSpec, MissingReviewError, build_condition_labels, capped_review_times

CFG = {
    "experiment_name": "time_budget",
    "data": {"kind": "synthetic", "synthetic": {"n_episodes": 60, "n_sessions": 3, "seed": 21}},
    "conditions": [{"name": "A", "policy": "full", "fraction": 1.0},
                   {"name": "T_rand", "policy": "random", "fraction": 0.25, "budget_unit": "seconds"},
                   {"name": "T_low", "policy": "low_confidence", "fraction": 0.25, "budget_unit": "seconds"},
                   {"name": "T_fake", "policy": "fake_success_priority", "fraction": 0.25, "budget_unit": "seconds"},
                   {"name": "E_low", "policy": "low_confidence", "fraction": 0.25}],
    "replicates": {"n": 2, "base_seed": 5},
    "bootstrap": {"n_boot": 100, "seed": 0},
}


@pytest.fixture(scope="module")
def setup():
    cfg = validate_config(copy.deepcopy(CFG))
    return cfg, prepare_data(cfg)


def _build(data, policy, unit="seconds", seed=0, reviews=None, frac=0.25):
    pool = data.ids("train", "val")
    return build_condition_labels(ConditionSpec("x", policy, frac, unit), pool, data.keyframes_by_episode,
                                  data.vlm_by_sample, reviews or data.pool_reviews, np.random.default_rng(seed), 120.0)


def test_time_budget_stops_when_time_runs_out(setup):
    cfg, data = setup
    times = capped_review_times(data.ids("train", "val"), data.pool_reviews, 120.0)
    for policy in ("random", "low_confidence", "fake_success_priority"):
        cl = _build(data, policy)
        budget, used = cl.meta["time_budget_s"], cl.meta["time_used_s"]
        sel = cl.selected_episodes
        assert sum(times[e] for e in sel[:-1]) < budget <= used + 1e-9 or len(sel) == len(times)
        assert used - times[sel[-1]] < budget  # only the last episode may overshoot
        assert cl.meta["realized_time_fraction"] == pytest.approx(used / sum(times.values()))
        assert cl.k == len(sel) and cl.meta["budget_unit"] == "seconds"


def test_equal_time_means_different_episode_counts(setup):
    cfg, data = setup
    n_eps = {p: _build(data, p).k for p in ("random", "low_confidence", "fake_success_priority")}
    fracs = {p: _build(data, p).meta["realized_time_fraction"] for p in n_eps}
    assert all(0.25 <= f < 0.25 + 0.1 for f in fracs.values()), fracs
    # harder (slower) episodes are picked first by confidence-based policies -> fewer episodes in the same time
    assert n_eps["low_confidence"] < n_eps["random"] or n_eps["fake_success_priority"] < n_eps["random"], n_eps


def test_time_budget_never_reads_labels(setup):
    cfg, data = setup
    corrupted = {eid: ReviewRecord(**{**r.__dict__, "labels": {s: Label.AMBIGUOUS for s in r.labels}})
                 for eid, r in data.pool_reviews.items()}
    for policy in ("random", "low_confidence", "fake_success_priority"):
        assert _build(data, policy, seed=3).selected_episodes == \
            _build(data, policy, seed=3, reviews=corrupted).selected_episodes


def test_time_budget_requires_times_and_valid_specs(setup):
    cfg, data = setup
    no_time = dict(data.pool_reviews)
    eid = next(iter(no_time))
    no_time[eid] = ReviewRecord(**{**no_time[eid].__dict__, "active_time_s": None, "timing_source": "missing"})
    with pytest.raises(MissingReviewError):
        _build(data, "random", reviews=no_time)
    _build(data, "random", unit="episodes", reviews=no_time)  # episode budgets do not need times
    with pytest.raises(ValueError):
        ConditionSpec("bad", "full", 1.0, "seconds")
    with pytest.raises(ValueError):
        ConditionSpec("bad", "random", 0.25, "minutes")
    bad = copy.deepcopy(CFG)
    bad["conditions"] = [{"name": "z", "policy": "none", "fraction": 0.0, "budget_unit": "seconds"}]
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_runner_reports_time_fields(setup):
    cfg, data = setup
    rows = run_conditions(data, cfg)
    by = {(r["condition"], r["replicate"]): r for r in rows}
    assert by[("T_rand", 0)]["budget"]["budget_unit"] == "seconds"
    assert "time_budget_s" in by[("T_rand", 0)]["budget"]
    assert by[("E_low", 0)]["budget"]["budget_unit"] == "episodes"
    assert by[("A", 0)]["budget"]["realized_time_fraction"] == pytest.approx(1.0)
    assert by[("T_rand", 0)]["selected_episodes"] != by[("T_rand", 1)]["selected_episodes"]
