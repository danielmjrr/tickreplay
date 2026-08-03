"""Leak-proof evaluation splits.

``purged_expanding_folds`` implements purged walk-forward cross-validation
(Lopez de Prado): expanding train windows, chronological test fifths, and a purge
that drops any training sample whose *label window* — not just its timestamp —
reaches into the test period, plus an embargo margin on top.

``holdout_cut`` supports the discipline of never touching a final period until one
pre-registered confirmation shot.
"""
import numpy as np


def holdout_cut(time_ns, holdout_start_ns):
    """Index of the first tick inside the holdout; data at [cut:] stays untouched."""
    return int(np.searchsorted(np.asarray(time_ns), np.int64(holdout_start_ns)))


def purged_expanding_folds(entry_time_ns, horizon_ns, embargo_ns, n_folds=5):
    """Yield (train_idx, test_idx) pairs. Fold k tests the k-th time slice and trains
    on everything before it minus samples whose window (t + horizon + embargo)
    reaches the test start. The first slice has no prior data and is never a test."""
    t = np.asarray(entry_time_ns, dtype="int64")
    lo, hi = t.min(), t.max() + 1
    edges = lo + (hi - lo) * np.arange(n_folds + 1) // n_folds
    folds = []
    for k in range(1, n_folds):
        test = np.flatnonzero((t >= edges[k]) & (t < edges[k + 1]))
        train = np.flatnonzero(t + horizon_ns + embargo_ns <= edges[k])
        if len(test) and len(train):
            folds.append((train, test))
    return folds
