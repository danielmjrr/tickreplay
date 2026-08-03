"""Synthetic ticks stand in for broker data, so their invariants have to hold."""
import numpy as np
import pandas as pd
import pytest

import tickreplay as tr

# 2024-01-01 is a Monday, so a 9-day window spans exactly one weekend
NINE_DAYS = dict(days=9, ticks_per_day=2000, start="2024-01-01", seed=2)


def test_time_never_goes_backwards():
    time_ns, _, _ = tr.synth.generate_ticks(days=5, ticks_per_day=3000, seed=1)
    assert len(time_ns) > 1000
    assert (np.diff(time_ns) >= 0).all()


def test_ask_is_always_above_bid():
    _, bid, ask = tr.synth.generate_ticks(days=5, ticks_per_day=3000, seed=1)
    assert len(bid) > 1000
    assert (ask > bid).all()


def test_no_ticks_land_on_a_weekend():
    time_ns, _, _ = tr.synth.generate_ticks(**NINE_DAYS)
    weekday = pd.DatetimeIndex(time_ns.astype("datetime64[ns]")).dayofweek
    assert set(weekday) == {0, 1, 2, 3, 4}


def test_day_count_covers_the_requested_window_minus_weekends():
    time_ns, _, _ = tr.synth.generate_ticks(**NINE_DAYS)
    days = np.unique(time_ns.astype("datetime64[ns]").astype("datetime64[D]"))
    expected = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
                         "2024-01-05", "2024-01-08", "2024-01-09"], "datetime64[D]")
    np.testing.assert_array_equal(days, expected)


def test_ticks_stay_inside_the_22h_session():
    time_ns, _, _ = tr.synth.generate_ticks(**NINE_DAYS)
    seconds_into_day = (time_ns % (86_400 * 10**9)) / 10**9
    assert seconds_into_day.max() < 22 * 3600
    assert seconds_into_day.max() > 20 * 3600, "session barely used -- fixture too thin"


def test_same_seed_reproduces_the_same_ticks():
    kwargs = dict(days=3, ticks_per_day=2000, seed=11)
    for first, second in zip(tr.synth.generate_ticks(**kwargs),
                             tr.synth.generate_ticks(**kwargs)):
        np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_ticks():
    _, bid_a, _ = tr.synth.generate_ticks(days=3, ticks_per_day=2000, seed=11)
    _, bid_b, _ = tr.synth.generate_ticks(days=3, ticks_per_day=2000, seed=12)
    assert len(bid_a) != len(bid_b) or not np.array_equal(bid_a, bid_b)


def test_weekend_only_window_raises_a_clear_error():
    with pytest.raises(ValueError, match="no weekdays"):
        tr.synth.generate_ticks(days=2, start="2024-01-06")  # Sat + Sun
