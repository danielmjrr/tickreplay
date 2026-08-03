"""Honest statistics for strategy evaluation.

Three tools that together keep a research program from fooling itself:

- a one-sided positive-expectancy test (the promotion hypothesis is "makes money";
  losing variants are avoid-knowledge, and letting their tiny two-sided p-values
  into a multiple-testing correction quietly loosens it for the winners),
- overlap thinning (labels from overlapping windows are correlated; a t-stat over
  them is inflated — test a greedy non-overlapping subset instead),
- Benjamini-Hochberg q-values over *every evaluation ever run* (see registry).
"""
import numpy as np
from scipy import stats as sps


def one_sided_positive(pnl):
    """(t_stat, p_value) for H1: mean > 0. Degenerate samples -> p = 1."""
    x = np.asarray(pnl, dtype="float64")
    if x.size < 2 or np.std(x, ddof=1) == 0:
        return 0.0, 1.0
    res = sps.ttest_1samp(x, 0.0, alternative="greater")
    return float(res.statistic), float(res.pvalue)


def thin_by_gap(entry_time_ns, min_gap_ns):
    """Boolean mask keeping a greedy subset of entries at least ``min_gap_ns`` apart."""
    t = np.asarray(entry_time_ns, dtype="int64")
    keep = np.zeros(t.size, dtype=bool)
    last = -np.int64(2**62)
    for i in range(t.size):
        if t[i] - last >= min_gap_ns:
            keep[i] = True
            last = t[i]
    return keep


def bh_qvalues(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype="float64")
    n = p.size
    order = np.argsort(p)
    q_sorted = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q
