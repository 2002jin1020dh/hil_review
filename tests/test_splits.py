import pytest

from hilreview.data.schema import DataKind, Episode, Keyframe, make_sample_id
from hilreview.data.splits import EXCLUDED, LeakageError, SplitSpec, assign_splits, check_split_integrity


def _episodes(n_per=7, sessions=("s1", "s2", "s3"), modes=("success_attempt", "near_miss", "clear_failure")):
    eps = []
    i = 0
    for s in sessions:
        for m in modes:
            for _ in range(n_per):
                eps.append(Episode(f"ep{i:03d}", s, m, 100, 10.0, DataKind.SYNTHETIC, source=f"src{i}"))
                i += 1
    return eps


def test_split_sizes_exact_and_no_leakage():
    eps = _episodes()  # 63 episodes
    spec = SplitSpec(audit_fraction=0.25, val_fraction_of_pool=0.2, seed=1)
    split = assign_splits(eps, spec)
    counts = check_split_integrity(eps, split)
    assert counts["audit"] == 16  # round_half_up(0.25 * 63 = 15.75)
    assert counts["val"] == 9  # round_half_up(0.2 * 47 = 9.4)
    assert counts["train"] == 63 - 16 - 9
    # every (session, mode) stratum contributes to the audit set
    strata = {(e.session_id, e.collection_mode) for e in eps if split[e.episode_id] == "audit"}
    assert len(strata) == 9


def test_split_deterministic_and_order_invariant():
    eps = _episodes()
    spec = SplitSpec(seed=7)
    a = assign_splits(eps, spec)
    b = assign_splits(list(reversed(eps)), spec)
    assert a == b
    c = assign_splits(eps, SplitSpec(seed=8))
    assert a != c  # different seed -> different (but equally valid) split


def test_ineligible_excluded():
    eps = _episodes(n_per=3)
    eps[0] = Episode("ep000", "s1", "success_attempt", 100, 10.0, DataKind.SYNTHETIC, eligible=False,
                     ineligible_reason="camera dropout")
    split = assign_splits(eps, SplitSpec(seed=3))
    assert split["ep000"] == EXCLUDED
    counts = check_split_integrity(eps, split)
    assert counts[EXCLUDED] == 1


def test_session_grouping_keeps_sessions_whole():
    eps = _episodes(n_per=2, sessions=tuple(f"s{i}" for i in range(8)))
    spec = SplitSpec(group_by="session_id", stratify_by=(), seed=5)
    split = assign_splits(eps, spec)
    check_split_integrity(eps, split, group_by="session_id")
    by_session = {}
    for e in eps:
        by_session.setdefault(e.session_id, set()).add(split[e.episode_id])
    assert all(len(v) == 1 for v in by_session.values())
    assert sum(1 for v in by_session.values() if v == {"audit"}) == 2  # round_half_up(0.25 * 8)


def test_session_grouping_rejects_inconsistent_strata():
    eps = _episodes(n_per=2)
    with pytest.raises(ValueError):
        assign_splits(eps, SplitSpec(group_by="session_id", stratify_by=("collection_mode",)))


def test_integrity_checks_catch_leakage():
    eps = _episodes(n_per=2)
    split = assign_splits(eps, SplitSpec(seed=2))
    # same underlying source registered twice in different splits
    audit_ep = next(e for e in eps if split[e.episode_id] == "audit")
    train_ep = next(e for e in eps if split[e.episode_id] == "train")
    dup = Episode(train_ep.episode_id, train_ep.session_id, train_ep.collection_mode, 100, 10.0,
                  DataKind.SYNTHETIC, source=audit_ep.source)
    eps2 = [dup if e.episode_id == train_ep.episode_id else e for e in eps]
    with pytest.raises(LeakageError):
        check_split_integrity(eps2, split)
    # session grouping violated
    with pytest.raises(LeakageError):
        check_split_integrity(eps, split, group_by="session_id")
    # duplicate sample ids / unknown episodes
    kf = Keyframe(make_sample_id(eps[0].episode_id, 0), eps[0].episode_id, 0, 0.0)
    with pytest.raises(LeakageError):
        check_split_integrity(eps, split, {eps[0].episode_id: [kf, kf]})
    with pytest.raises(LeakageError):
        check_split_integrity(eps, split, {"nope": []})
    # ineligible assigned to a real split
    bad = dict(split)
    eps3 = list(eps)
    eps3[0] = Episode(eps[0].episode_id, eps[0].session_id, eps[0].collection_mode, 100, 10.0,
                      DataKind.SYNTHETIC, eligible=False, ineligible_reason="x")
    with pytest.raises(LeakageError):
        check_split_integrity(eps3, bad)
