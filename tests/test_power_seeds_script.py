"""Checks for scripts/power_seeds.py (planning estimate for the D-30 online non-inferiority rule)."""

import importlib.util
import math
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "power_seeds.py"
_spec = importlib.util.spec_from_file_location("power_seeds", _PATH)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


@pytest.mark.parametrize("q,df,expected", [
    (0.95, 2, 2.919986), (0.95, 4, 2.131847), (0.95, 8, 1.859548),
    (0.975, 10, 2.228139), (0.95, 1e7, 1.644854),
])
def test_t_ppf_matches_tables(q, df, expected):
    assert ps.t_ppf(q, df) == pytest.approx(expected, abs=2e-5)


def test_t_cdf_symmetry_and_fractional_df():
    assert ps.t_cdf(0.0, 5) == pytest.approx(0.5)
    assert ps.t_cdf(-1.3, 3.7) == pytest.approx(1 - ps.t_cdf(1.3, 3.7), abs=1e-12)
    q45 = ps.t_ppf(0.95, 4.5)  # Welch df is usually fractional
    assert ps.t_ppf(0.95, 5) < q45 < ps.t_ppf(0.95, 4)


def test_welch_known_values():
    diff, se, df = ps.welch([0.7, 0.8, 0.75], [0.7, 0.8, 0.75])
    assert diff == pytest.approx(0.0)
    assert se == pytest.approx(math.sqrt(2 * 0.0025 / 3))
    assert df == pytest.approx(4.0)  # equal n and variances -> 2n - 2
    with pytest.raises(ValueError):
        ps.welch([0.5], [0.5, 0.6])


def test_welch_noninferior_decisions():
    same = [0.7, 0.8, 0.75]
    assert ps.welch_noninferior(same, same, delta=0.20)            # tight and equal -> non-inferior
    assert not ps.welch_noninferior([0.2, 0.3, 0.25], same, 0.20)  # clearly worse -> not
    assert not ps.welch_noninferior([0.3, 0.9, 0.6], [0.2, 0.95, 0.7], 0.20)  # noisy -> inconclusive
    assert ps.welch_noninferior([0.6, 0.6], [0.7, 0.7], 0.20)      # zero variance: decide on margin
    assert not ps.welch_noninferior([0.4, 0.4], [0.7, 0.7], 0.20)


def test_simulate_is_deterministic_and_sane():
    r1 = ps.simulate(0.7, 0.10, 3, 20, 0.20, sims=300, rng_seed=1)
    r2 = ps.simulate(0.7, 0.10, 3, 20, 0.20, sims=300, rng_seed=1)
    assert r1 == r2
    easy = ps.simulate(0.7, 0.0, 10, 200, 0.50, sims=200, rng_seed=2)  # huge margin -> ~always
    assert easy[0] > 0.99 and easy[1] > 0.99
    edge = ps.simulate(0.7, 0.10, 5, 20, 0.0, sims=600, rng_seed=3)    # delta 0 -> type I error ~5%
    assert edge[0] < 0.12 and edge[1] < 0.12
