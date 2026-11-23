"""D-30 online non-inferiority (Welch t, per-seed success rates)."""

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

from hilreview.data.schema import DataKind
from hilreview.evaluation.noninferiority import noninferiority, t_cdf, t_ppf, welch
from hilreview.evaluation.online import OnlineEpisodeRecord, RunRecord, compare_to_reference, summarize_conditions

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "power_seeds.py"
_spec = importlib.util.spec_from_file_location("power_seeds_for_d30", _PATH)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


@pytest.mark.parametrize("df,expected", [
    (1, 6.313752), (2, 2.919986), (4, 2.131847), (10, 1.812461), (30, 1.697261), (math.inf, 1.644854),
])
def test_t_ppf_095_matches_tables(df, expected):
    assert t_ppf(0.95, df) == pytest.approx(expected, abs=2e-5)


def test_t_cdf_basic():
    assert t_cdf(0.0, 3) == pytest.approx(0.5)
    assert t_cdf(-1.3, 5) == pytest.approx(1 - t_cdf(1.3, 5))
    assert t_ppf(0.95, 2) > t_ppf(0.95, 2.5) > t_ppf(0.95, 3)  # fractional df lies between


def test_welch_hand_computed():
    diff, se, df = welch([0.8, 0.7, 0.9], [0.9, 0.85, 0.95])
    vx, va = 0.01 / 3, 0.0025 / 3
    assert diff == pytest.approx(-0.1)
    assert se == pytest.approx(math.sqrt(vx + va))
    assert df == pytest.approx((vx + va) ** 2 / (vx ** 2 / 2 + va ** 2 / 2))


def test_decisions():
    r = noninferiority([0.8, 0.7, 0.9], [0.9, 0.85, 0.95])
    assert r["decision"] == "inconclusive"
    assert r["lower"] == pytest.approx(r["diff"] - t_ppf(0.95, r["df"]) * r["se"])
    assert r["lower"] < -0.2 < r["upper"]
    assert noninferiority([0.9, 0.95, 0.9, 0.95], [0.9, 0.9, 0.95, 0.95])["decision"] == "noninferior"
    assert noninferiority([0.2, 0.25, 0.3], [0.9, 0.95, 0.85])["decision"] == "inferior"


def test_zero_variance_and_strict_boundary():
    assert noninferiority([1.0, 1.0], [1.0, 1.0])["decision"] == "noninferior"
    assert noninferiority([0.5, 0.5], [0.9, 0.9])["decision"] == "inferior"
    # diff exactly -delta: neither "lower > -delta" nor "upper < -delta" -> inconclusive
    edge = noninferiority([0.7, 0.7], [0.9, 0.9], delta=0.2)
    assert edge["diff"] == pytest.approx(-0.2)
    edge2 = noninferiority([0.5, 0.5], [0.75, 0.75], delta=0.25)  # exact in binary
    assert edge2["decision"] == "inconclusive"
    # one arm with zero variance still gives a finite df carried by the other arm
    r = noninferiority([0.8, 0.9, 1.0], [0.9, 0.9, 0.9])
    assert math.isfinite(r["df"]) and r["df"] == pytest.approx(2.0)


def test_insufficient_and_invalid():
    assert noninferiority([0.9], [0.9, 0.8])["decision"] == "insufficient_runs"
    with pytest.raises(ValueError):
        noninferiority([0.9, 1.2], [0.9, 0.8])
    with pytest.raises(ValueError):
        noninferiority([0.9, None], [0.9, 0.8])
    with pytest.raises(ValueError):
        noninferiority([0.9, 0.8], [0.9, 0.8], delta=0)


def test_agrees_with_planning_script():
    rng = random.Random(0)
    for _ in range(300):
        nx, na = rng.randint(2, 6), rng.randint(2, 6)
        x = [rng.randint(0, 20) / 20 for _ in range(nx)]
        a = [rng.randint(0, 20) / 20 for _ in range(na)]
        if len(set(x)) == 1 and len(set(a)) == 1:
            continue
        ours = noninferiority(x, a, delta=0.2)
        assert (ours["decision"] == "noninferior") == ps.welch_noninferior(x, a, 0.2)
        d1, s1, f1 = welch(x, a)
        d2, s2, f2 = ps.welch(x, a)
        assert (d1, s1) == pytest.approx((d2, s2)) and (f1 == f2 or f1 == pytest.approx(f2))


def _run_eps(run_id, n_success, n_eval=10):
    return [OnlineEpisodeRecord(run_id, "eval", i, "classifier_success" if i < n_success else "timeout",
                                i < n_success, DataKind.SYNTHETIC) for i in range(n_eval)]


@pytest.fixture
def online_logs():
    key = {"r1": {"condition": "A", "replicate": 0}, "r2": {"condition": "A", "replicate": 1},
           "r3": {"condition": "A", "replicate": 2}, "r4": {"condition": "C", "replicate": 0},
           "r5": {"condition": "C", "replicate": 1}, "r6": {"condition": "C", "replicate": 2},
           "r7": {"condition": "C", "replicate": 3}}
    successes = {"r1": 9, "r2": 8, "r3": 10, "r4": 9, "r5": 9, "r6": 10}
    eps = [e for r, k in successes.items() for e in _run_eps(r, k)]
    runs = [RunRecord(r, "completed", DataKind.SYNTHETIC) for r in successes] + [
        RunRecord("r7", "crashed", DataKind.SYNTHETIC)]
    return eps, runs, key


def test_compare_to_reference_primary_and_sensitivity(online_logs):
    eps, runs, key = online_logs
    comp = compare_to_reference(summarize_conditions(eps, runs, key), "A")
    c = comp["C"]
    assert c["primary"]["n_x"] == 3 and c["primary"]["mean_x"] == pytest.approx(28 / 30)
    assert c["primary"]["decision"] == "noninferior"
    s = c["sensitivity_failed_runs_as_zero"]
    assert s["n_x"] == 4 and s["mean_x"] == pytest.approx(2.8 / 4)
    assert s["decision"] != "noninferior"  # the crashed run (as 0) widens the interval a lot
    assert c["failed_runs"] == {"x": 1, "reference": 0}
    assert c["pooled_wilson_descriptive"]["diff"] == pytest.approx(28 / 30 - 27 / 30)
    with pytest.raises(ValueError):
        compare_to_reference(summarize_conditions(eps, runs, key), "B")


def test_cli_online_summary_with_reference(online_logs, tmp_path, capsys):
    from hilreview.cli import main
    from hilreview.data.io import write_jsonl

    eps, runs, key = online_logs
    write_jsonl(tmp_path / "eps.jsonl", eps)
    write_jsonl(tmp_path / "runs.jsonl", runs)
    (tmp_path / "key.json").write_text(json.dumps([{"run_id": r, **k} for r, k in key.items()]))
    assert main(["online-summary", "--episodes", str(tmp_path / "eps.jsonl"), "--runs", str(tmp_path / "runs.jsonl"),
                 "--key", str(tmp_path / "key.json"), "--out", str(tmp_path / "s.json"), "--reference", "A"]) == 0
    out = json.loads((tmp_path / "s.json").read_text())
    assert out["comparisons_D30"]["C"]["primary"]["decision"] == "noninferior"
    assert "C − A: 비열등" in capsys.readouterr().out
