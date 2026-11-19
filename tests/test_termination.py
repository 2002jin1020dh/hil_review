import pytest

from hilreview.evaluation.termination import classify_episode, first_trigger, replay_all, summarize_replays


def test_first_trigger_strict_and_debounce():
    s = [0.1, 0.6, 0.4, 0.7, 0.8]
    assert first_trigger(s, 0.5) == 1
    assert first_trigger(s, 0.5, debounce_k=2) == 4
    assert first_trigger([0.5, 0.5], 0.5) is None  # LeRobot uses strict '>'
    assert first_trigger([0.9, None, 0.9], 0.5, debounce_k=2) is None  # None resets streak
    with pytest.raises(ValueError):
        first_trigger(s, 0.5, debounce_k=0)


@pytest.mark.parametrize(
    "states,scores,tol,expected",
    [
        ("FFSS", [0.1, 0.2, 0.9, 0.9], 0, "true_trigger"),
        ("FFSS", [0.9, 0.1, 0.1, 0.1], 0, "early_trigger"),
        ("FFFF", [0.1, 0.9, 0.1, 0.1], 0, "false_trigger"),
        ("FUFF", [0.1, 0.9, 0.1, 0.1], 0, "ambiguous_trigger"),
        ("FFSS", [0.1, 0.1, 0.1, 0.1], 0, "missed_success"),
        ("FFF", [0.1, 0.1, 0.1], 0, "true_no_trigger"),
        ("FFS", [0.1, 0.9, 0.1], 0, "early_trigger"),
        ("FFS", [0.1, 0.9, 0.1], 1, "true_trigger"),  # within one keyframe of onset
        ("FSFF", [0.1, 0.1, 0.1, 0.9], 0, "false_trigger"),  # success then revert; trigger after revert
    ],
)
def test_classify_episode_categories(states, scores, tol, expected):
    r = classify_episode("e", scores, list(states), threshold=0.5, tolerance=tol)
    assert r.outcome == expected


def test_summary_rates_and_denominators():
    eps = {
        "a": ("FFSS", [0.1, 0.1, 0.9, 0.9]),  # true_trigger, delay 0
        "b": ("FFSS", [0.9, 0.1, 0.1, 0.1]),  # early
        "c": ("FFFF", [0.1, 0.9, 0.1, 0.1]),  # false
        "d": ("FUFF", [0.1, 0.9, 0.1, 0.1]),  # ambiguous (excluded)
        "e": ("FFSS", [0.1, 0.1, 0.1, 0.1]),  # missed
    }
    reps = replay_all({k: v[1] for k, v in eps.items()}, {k: list(v[0]) for k, v in eps.items()}, 0.5)
    s = summarize_replays(reps)
    assert s["n_episodes"] == 5 and s["n_ambiguous_trigger_excluded"] == 1
    assert s["fst_count"] == 2
    assert s["fst_rate"] == pytest.approx(2 / 4)
    assert s["fst_share_of_triggers"] == pytest.approx(2 / 3)
    assert s["missed_success_rate"] == pytest.approx(1 / 3)  # a, b, e reach success; e missed
    assert s["mean_trigger_delay_keyframes"] == 0


def test_replay_requires_scores_for_every_episode():
    with pytest.raises(ValueError):
        replay_all({}, {"x": ["F"]}, 0.5)
    with pytest.raises(ValueError):
        classify_episode("x", [0.1], ["Q"], 0.5)
