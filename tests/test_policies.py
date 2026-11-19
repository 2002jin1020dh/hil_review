import numpy as np
import pytest

from hilreview.data.schema import Label
from hilreview.review.policies import (
    EpisodeVLMView,
    episode_uncertainty,
    fake_success_key,
    review_budget,
    select_episodes,
)

S, F, A = Label.SUCCESS, Label.FAILURE, Label.AMBIGUOUS


@pytest.mark.parametrize("fraction,n,k", [(0.25, 88, 22), (0.25, 75, 19), (0.25, 10, 3), (0.25, 2, 1),
                                          (0.0, 50, 0), (1.0, 50, 50), (0.25, 0, 0), (0.5, 5, 3)])
def test_budget_round_half_up(fraction, n, k):
    assert review_budget(fraction, n) == k


def test_budget_rejects_bad_fraction():
    with pytest.raises(ValueError):
        review_budget(1.5, 10)


def _views():
    return [
        EpisodeVLMView("e1", (F, F, F), (0.05, 0.02, 0.10)),  # confident failure
        EpisodeVLMView("e2", (F, S, S), (0.20, 0.55, 0.60)),  # weak success claim, uncertain
        EpisodeVLMView("e3", (F, S, S), (0.01, 0.99, 0.98)),  # confident success claim
        EpisodeVLMView("e4", (F, F, A), (0.30, 0.40, 0.50)),  # unsure answer -> max uncertainty
        EpisodeVLMView("e5", (F, None, F), (0.02, None, 0.03)),  # VLM error -> max uncertainty
    ]


def test_uncertainty_and_fake_success_keys():
    v = {x.episode_id: x for x in _views()}
    # sum over keyframes of 1 - |2p - 1|; unusable labels (None / ambiguous) count 1
    assert episode_uncertainty(v["e1"]) == pytest.approx(0.1 + 0.04 + 0.2)
    assert episode_uncertainty(v["e2"]) == pytest.approx(0.4 + 0.9 + 0.8)
    assert episode_uncertainty(v["e4"]) == pytest.approx(0.6 + 0.8 + 1.0)
    assert episode_uncertainty(v["e5"]) == pytest.approx(0.04 + 1.0 + 0.06)
    assert fake_success_key(v["e2"]) == (True, pytest.approx(0.45 + 0.40))  # sum of (1 - p) over claimed keyframes
    assert fake_success_key(v["e3"]) == (True, pytest.approx(0.01 + 0.02))
    assert fake_success_key(v["e1"]) == (False, 0.0)


def test_low_confidence_order():
    sel = select_episodes("low_confidence", _views(), 3, np.random.default_rng(0))
    assert sel == ["e4", "e2", "e5"]  # 2.4 > 2.1 > 1.1 > 0.34 > 0.08


def test_fake_success_priority_order():
    sel = select_episodes("fake_success_priority", _views(), 4, np.random.default_rng(0))
    assert sel == ["e2", "e3", "e4", "e5"]  # claimed successes (most expected fake labels first), then low confidence
    # a long episode with many moderately confident claims outranks one weak claim
    many = EpisodeVLMView("e6", (S,) * 6, (0.8,) * 6)  # sum(1 - p) = 1.2
    one = EpisodeVLMView("e7", (F, S), (0.1, 0.4))  # 0.6
    assert select_episodes("fake_success_priority", [one, many], 1, np.random.default_rng(0)) == ["e6"]


def test_random_policy_deterministic_and_order_invariant():
    v = _views()
    a = select_episodes("random", v, 2, np.random.default_rng(42))
    b = select_episodes("random", list(reversed(v)), 2, np.random.default_rng(42))
    assert a == b and len(set(a)) == 2
    picks = {tuple(sorted(select_episodes("random", v, 2, np.random.default_rng(s)))) for s in range(30)}
    assert len(picks) > 3  # actually random across seeds


def test_full_none_and_errors():
    v = _views()
    assert select_episodes("full", v, 0, np.random.default_rng(0)) == ["e1", "e2", "e3", "e4", "e5"]
    assert select_episodes("none", v, 5, np.random.default_rng(0)) == []
    with pytest.raises(ValueError):
        select_episodes("oracle", v, 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        select_episodes("random", v, 6, np.random.default_rng(0))
    with pytest.raises(ValueError):
        select_episodes("random", v + v[:1], 1, np.random.default_rng(0))
