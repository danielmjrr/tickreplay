import numpy as np
import pytest

from tickreplay.research.bocpd import bocpd_features

HAZARD = 1 / 250


def synth(seed=3):
    """Three regimes: a mean shift at 300 and a volatility shift at 600."""
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0, 1, 300), rng.normal(4, 1, 300), rng.normal(4, 5, 300)]


def test_detects_planted_mean_and_volatility_shifts():
    p = bocpd_features(synth(), hazard=HAZARD)["p_cp_10"]

    assert p[302:310].max() > 0.5
    assert p[602:612].max() > 0.5
    assert p[200:290].mean() < 0.1
    assert p[500:590].mean() < 0.1


def test_expected_run_length_collapses_at_a_break():
    erl = bocpd_features(synth(), hazard=HAZARD)["erl"]

    assert erl[290] > 100
    assert erl[310] < erl[290] / 2
    assert erl[590] > 100
    assert erl[610] < erl[590] / 2


def test_run_length_grows_inside_a_stable_segment():
    erl = bocpd_features(synth(), hazard=HAZARD)["erl"]

    assert erl[100] < erl[200] < erl[290]


def test_is_strictly_causal():
    x = synth()
    baseline = bocpd_features(x, hazard=HAZARD)
    mutated = x.copy()
    mutated[400:] = 99.0
    after = bocpd_features(mutated, hazard=HAZARD)

    np.testing.assert_array_equal(baseline["p_cp_10"][:400], after["p_cp_10"][:400])
    np.testing.assert_array_equal(baseline["erl"][:400], after["erl"][:400])


def test_outputs_are_valid_probabilities_and_bounded_run_lengths():
    out = bocpd_features(synth(), hazard=HAZARD)
    p, erl = out["p_cp_10"], out["erl"]

    assert p.shape == erl.shape == (900,)
    assert ((p >= 0) & (p <= 1)).all()
    assert (erl >= 0).all()
    assert (erl <= np.arange(1, len(erl) + 1)).all()
    assert np.isfinite(p).all()
    assert np.isfinite(erl).all()


def test_first_observation_is_always_a_fresh_run():
    out = bocpd_features(np.array([0.0, 0.1, -0.2, 0.05]), hazard=HAZARD)

    assert out["p_cp_10"][0] == 1.0
    # a single observation can only be run length 0 or 1, weighted by the hazard
    assert out["erl"][0] == pytest.approx(1 - HAZARD)
