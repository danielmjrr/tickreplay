import numpy as np

from tickreplay.research.validation import holdout_cut, purged_expanding_folds

HOUR_NS = 3600 * 10**9
DAY_NS = 24 * HOUR_NS


def sample_times(n=6000, step_ns=37 * 60 * 10**9):
    return 1_700_000_000 * 10**9 + np.arange(n, dtype="int64") * step_ns


def slice_edges(t, n_folds):
    lo, hi = t.min(), t.max() + 1
    return lo + (hi - lo) * np.arange(n_folds + 1) // n_folds


def test_purged_folds_never_let_a_train_window_reach_the_test_start():
    t = sample_times()
    horizon, embargo = np.int64(HOUR_NS), np.int64(DAY_NS)
    edges = slice_edges(t, 5)

    folds = purged_expanding_folds(t, horizon, embargo, n_folds=5)

    assert len(folds) == 4
    for k, (train, test) in enumerate(folds, start=1):
        assert len(train) > 0
        assert len(test) > 0
        assert (t[train] + horizon + embargo <= edges[k]).all()
        assert (t[train] + horizon + embargo <= t[test].min()).all()


def test_purged_folds_train_strictly_before_test():
    t = sample_times()

    folds = purged_expanding_folds(t, np.int64(HOUR_NS), np.int64(DAY_NS), n_folds=5)

    for train, test in folds:
        assert t[train].max() < t[test].min()
        assert not set(train.tolist()) & set(test.tolist())


def test_purging_drops_samples_a_plain_split_would_have_kept():
    t = sample_times()
    edges = slice_edges(t, 5)

    unpurged = purged_expanding_folds(t, np.int64(0), np.int64(0), n_folds=5)
    purged = purged_expanding_folds(t, np.int64(HOUR_NS), np.int64(DAY_NS), n_folds=5)

    for (train_u, _), (train_p, _) in zip(unpurged, purged):
        assert len(train_p) < len(train_u)
        assert set(train_p.tolist()) < set(train_u.tolist())
    assert (t[unpurged[0][0]].max() < edges[1])


def test_first_slice_is_never_a_test_fold():
    t = sample_times()
    edges = slice_edges(t, 5)

    folds = purged_expanding_folds(t, np.int64(0), np.int64(0), n_folds=5)

    for _, test in folds:
        assert t[test].min() >= edges[1]


def test_test_folds_partition_the_remaining_time_into_equal_slices():
    t = sample_times(n=8760, step_ns=HOUR_NS)

    folds = purged_expanding_folds(t, np.int64(0), np.int64(0), n_folds=5)

    sizes = [len(test) for _, test in folds]
    assert max(sizes) - min(sizes) <= 0.2 * max(sizes)
    covered = np.concatenate([test for _, test in folds])
    assert len(covered) == len(set(covered.tolist()))
    edges = slice_edges(t, 5)
    assert len(covered) == int((t >= edges[1]).sum())


def test_purged_folds_honour_n_folds():
    t = sample_times()

    assert len(purged_expanding_folds(t, np.int64(0), np.int64(0), n_folds=3)) == 2
    assert len(purged_expanding_folds(t, np.int64(0), np.int64(0), n_folds=10)) == 9


def test_holdout_cut_returns_first_index_at_or_after_the_boundary():
    t = np.arange(10, dtype="int64") * 1000

    assert holdout_cut(t, 5000) == 5
    assert holdout_cut(t, 4999) == 5
    assert holdout_cut(t, 5001) == 6
    assert holdout_cut(t, 0) == 0
    assert holdout_cut(t, -1) == 0
    assert holdout_cut(t, 10**9) == len(t)


def test_holdout_cut_splits_cleanly_around_the_boundary():
    t = sample_times()
    boundary = int(t[0] + (t[-1] - t[0]) * 0.8)

    cut = holdout_cut(t, boundary)

    assert (t[:cut] < boundary).all()
    assert (t[cut:] >= boundary).all()
