import numpy as np
import pytest
from scipy import stats as sps

from tickreplay.research.stats import bh_qvalues, one_sided_positive, thin_by_gap


def test_one_sided_positive_matches_scipy():
    sample = np.array([0.4, -0.1, 0.9, 0.2, -0.3, 1.1, 0.05, 0.6, -0.2, 0.35])
    ref = sps.ttest_1samp(sample, 0.0, alternative="greater")

    t_stat, p_value = one_sided_positive(sample)

    assert t_stat == pytest.approx(float(ref.statistic))
    assert p_value == pytest.approx(float(ref.pvalue))


def test_one_sided_positive_is_one_sided():
    rng = np.random.default_rng(0)
    winners = rng.normal(0.4, 1.0, 400)
    losers = -winners

    assert one_sided_positive(winners)[1] < 0.01
    assert one_sided_positive(losers)[1] > 0.99


def test_one_sided_positive_rejects_degenerate_samples():
    assert one_sided_positive([]) == (0.0, 1.0)
    assert one_sided_positive([1.0]) == (0.0, 1.0)
    assert one_sided_positive([3.0, 3.0, 3.0, 3.0]) == (0.0, 1.0)
    assert one_sided_positive([0.0, 0.0]) == (0.0, 1.0)


def test_thin_by_gap_keeps_greedy_non_overlapping_subset():
    times = np.array([0, 5, 9, 10, 25, 26, 40], dtype="int64")

    keep = thin_by_gap(times, 10)

    np.testing.assert_array_equal(keep, [True, False, False, True, True, False, True])
    assert (np.diff(times[keep]) >= 10).all()


def test_thin_by_gap_edges():
    times = np.array([0, 1, 2, 3], dtype="int64")

    assert thin_by_gap(times, 0).all()
    np.testing.assert_array_equal(thin_by_gap(times, 10**9),
                                  [True, False, False, False])
    assert thin_by_gap(np.array([], dtype="int64"), 10).size == 0


def test_bh_qvalues_matches_hand_computed_example():
    # raw = p * n / rank -> [.006, .024, .078, .0615, .0504, .06]; cumulative min
    # from the largest p downwards enforces monotonicity.
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060])

    q = bh_qvalues(pvals)

    np.testing.assert_allclose(q, [0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.060])


def test_bh_qvalues_independent_of_input_order():
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060])
    perm = np.array([3, 0, 5, 2, 4, 1])

    q = bh_qvalues(pvals)
    q_perm = bh_qvalues(pvals[perm])

    np.testing.assert_allclose(q_perm, q[perm])


def test_bh_qvalues_are_monotone_and_bounded():
    rng = np.random.default_rng(7)
    pvals = np.r_[rng.uniform(0, 1, 200), rng.uniform(0, 1e-4, 20), 1.0, 0.0]

    q = bh_qvalues(pvals)

    assert ((q >= 0) & (q <= 1)).all()
    order = np.argsort(pvals)
    assert (np.diff(q[order]) >= -1e-12).all()
    assert (q >= pvals - 1e-12).all()


def test_bh_qvalues_single_value_is_unchanged():
    np.testing.assert_allclose(bh_qvalues([0.03]), [0.03])
