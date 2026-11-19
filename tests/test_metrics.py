import math

import pytest

from hilreview.evaluation.metrics import (
    BinaryCounts,
    binary_rates,
    cluster_bootstrap_ci,
    cohen_kappa,
    count_binary,
    summarize_values,
    wilson_interval,
)


def test_count_binary_known_case_and_exclusions():
    r = count_binary([1, 1, 0, 0, None, 1], [1, 0, 1, 0, 1, None])
    assert r.counts == BinaryCounts(tp=1, fp=1, tn=1, fn=1)
    assert (r.n_ref_missing, r.n_pred_missing, r.n_pairs) == (1, 1, 6)
    rates = binary_rates(r.counts)
    for k in ("accuracy", "precision", "recall", "specificity", "fpr", "fnr", "f1", "balanced_accuracy"):
        assert rates[k] == pytest.approx(0.5), k


def test_rates_hand_computed():
    c = BinaryCounts(tp=30, fp=5, tn=55, fn=10)
    r = binary_rates(c)
    assert r["accuracy"] == pytest.approx(85 / 100)
    assert r["precision"] == pytest.approx(30 / 35)
    assert r["recall"] == pytest.approx(30 / 40)
    assert r["fpr"] == pytest.approx(5 / 60)
    assert r["fnr"] == pytest.approx(10 / 40)
    assert r["specificity"] == pytest.approx(55 / 60)
    p, rc = 30 / 35, 30 / 40
    assert r["f1"] == pytest.approx(2 * p * rc / (p + rc))


def test_undefined_rates_are_none_not_zero():
    r = binary_rates(BinaryCounts(tp=0, fp=0, tn=10, fn=0))  # no positives anywhere
    assert r["precision"] is None and r["recall"] is None and r["fnr"] is None
    assert r["fpr"] == 0.0 and r["accuracy"] == 1.0
    assert binary_rates(BinaryCounts())["accuracy"] is None


def test_count_binary_rejects_bad_values():
    with pytest.raises(ValueError):
        count_binary([2], [1])
    with pytest.raises(ValueError):
        count_binary([1, 0], [1])


def test_wilson_reference_values():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(0, 0) is None


def test_cluster_bootstrap_deterministic_and_sane():
    clusters = [BinaryCounts(tp=3, fp=1, tn=5, fn=1) for _ in range(10)] + \
               [BinaryCounts(tp=1, fp=3, tn=4, fn=2) for _ in range(10)]
    a = cluster_bootstrap_ci(clusters, "accuracy", n_boot=500, seed=1)
    b = cluster_bootstrap_ci(clusters, "accuracy", n_boot=500, seed=1)
    assert a == b
    total = sum(clusters, BinaryCounts())
    point = binary_rates(total)["accuracy"]
    assert a["lo"] <= point <= a["hi"]
    # identical clusters -> zero-width interval
    same = cluster_bootstrap_ci([BinaryCounts(1, 1, 1, 1)] * 5, "accuracy", n_boot=200, seed=0)
    assert same["lo"] == pytest.approx(0.5) and same["hi"] == pytest.approx(0.5)
    # metric undefined in every resample -> no interval claimed
    none = cluster_bootstrap_ci([BinaryCounts(0, 0, 5, 0)] * 5, "recall", n_boot=100, seed=0)
    assert none["lo"] is None and none["n_valid"] == 0
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(clusters, "nonsense")


def test_cohen_kappa_known_value():
    k = cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0])
    assert k["observed_agreement"] == pytest.approx(0.75)
    assert k["kappa"] == pytest.approx(0.5)
    assert cohen_kappa([None], [1])["kappa"] is None
    assert cohen_kappa(["a", "a"], ["a", "a"])["kappa"] is None  # pe == 1 -> undefined


def test_summarize_values():
    s = summarize_values([1.0, 2.0, 3.0, None])
    assert s["mean"] == 2.0 and s["sd"] == pytest.approx(1.0) and s["n"] == 3 and s["n_missing"] == 1
    assert summarize_values([None])["mean"] is None
    assert summarize_values([5.0])["sd"] is None
    assert not math.isnan(summarize_values([1.0, 1.0])["sd"])
