import numpy as np
import pytest

from tickreplay.research.labels import excursions, first_passage


def random_ticks(seed, n=5000):
    rng = np.random.default_rng(seed)
    time_ns = np.cumsum(rng.integers(10**8, 2 * 10**9, n)).astype("int64")
    bid = 100 + np.cumsum(rng.normal(0, 0.05, n))
    ask = bid + rng.uniform(0.01, 0.05, n)
    return time_ns, bid, ask


def oracle_excursions(time_ns, bid, ask, entry_tick, d, checkpoints_ns):
    """Running MFE/MAE and signed close-out return at each checkpoint after entry."""
    entry_price = ask[entry_tick] if d == 1 else bid[entry_tick]
    t0 = time_ns[entry_tick]
    out = []
    mfe = mae = last_pnl = 0.0
    j = entry_tick + 1
    for cp in checkpoints_ns:
        while j < len(time_ns) and time_ns[j] <= t0 + cp:
            last_pnl = bid[j] - entry_price if d == 1 else entry_price - ask[j]
            mfe = max(mfe, last_pnl)
            mae = max(mae, -last_pnl)
            j += 1
        out.append((mfe, mae, last_pnl))
    return entry_price, out


def oracle_first_passage(time_ns, bid, ask, entry_tick, d, target, stop, horizon_ns):
    entry_price = ask[entry_tick] if d == 1 else bid[entry_tick]
    t0 = time_ns[entry_tick]
    pnl = 0.0
    j = entry_tick + 1
    while j < len(time_ns) and time_ns[j] <= t0 + horizon_ns:
        pnl = bid[j] - entry_price if d == 1 else entry_price - ask[j]
        if pnl >= target:
            return 1, pnl, time_ns[j]
        if -pnl >= stop:
            return -1, pnl, time_ns[j]
        j += 1
    return 0, pnl, min(t0 + horizon_ns, time_ns[-1])


def test_excursions_match_bruteforce_oracle():
    time_ns, bid, ask = random_ticks(9)
    rng = np.random.default_rng(9)
    entries = np.sort(rng.choice(np.arange(1, len(time_ns) - 500), 40, replace=False))
    entries = entries.astype("int64")
    dirs = rng.choice(np.array([-1, 1], dtype="int8"), 40)
    cps = np.array([60, 300, 900], dtype="int64") * 10**9

    out = excursions(time_ns, bid, ask, entries, dirs, cps)

    assert out["mfe"].shape == (40, 3)
    assert out["mae"].shape == (40, 3)
    assert out["ret"].shape == (40, 3)
    for k in range(40):
        ep, expected = oracle_excursions(time_ns, bid, ask, entries[k], dirs[k], cps)
        assert out["entry_price"][k] == pytest.approx(ep)
        for ci, (mfe, mae, ret) in enumerate(expected):
            assert out["mfe"][k, ci] == pytest.approx(mfe, abs=1e-12), (k, ci)
            assert out["mae"][k, ci] == pytest.approx(mae, abs=1e-12), (k, ci)
            assert out["ret"][k, ci] == pytest.approx(ret, abs=1e-12), (k, ci)


def test_excursions_enter_at_ask_for_long_and_bid_for_short():
    time_ns = np.arange(3, dtype="int64") * 10**9
    bid = np.array([100.0, 100.0, 100.0])
    ask = bid + 0.4
    entries = np.array([0, 0], dtype="int64")
    dirs = np.array([1, -1], dtype="int8")
    cps = np.array([10 * 10**9], dtype="int64")

    out = excursions(time_ns, bid, ask, entries, dirs, cps)

    assert out["entry_price"][0] == pytest.approx(100.4)
    assert out["entry_price"][1] == pytest.approx(100.0)
    # marking to the opposite side: a flat market still costs the spread both ways
    assert out["ret"][0, 0] == pytest.approx(-0.4)
    assert out["ret"][1, 0] == pytest.approx(-0.4)
    assert out["mae"][0, 0] == pytest.approx(0.4)
    assert out["mae"][1, 0] == pytest.approx(0.4)
    assert out["mfe"][0, 0] == 0.0
    assert out["mfe"][1, 0] == 0.0


def test_excursions_are_monotone_across_checkpoints():
    time_ns, bid, ask = random_ticks(21, n=2000)
    entries = np.array([10, 500, 900], dtype="int64")
    dirs = np.array([1, -1, 1], dtype="int8")
    cps = np.array([60, 300, 900], dtype="int64") * 10**9

    out = excursions(time_ns, bid, ask, entries, dirs, cps)

    assert (np.diff(out["mfe"], axis=1) >= 0).all()
    assert (np.diff(out["mae"], axis=1) >= 0).all()


def test_entry_at_last_tick_gives_zero_excursions():
    time_ns = np.arange(5, dtype="int64") * 10**9
    bid = np.full(5, 10.0)
    ask = bid + 0.1

    out = excursions(time_ns, bid, ask, np.array([4], dtype="int64"),
                     np.array([1], dtype="int8"), np.array([10**9], dtype="int64"))

    assert out["mfe"][0, 0] == 0.0
    assert out["mae"][0, 0] == 0.0
    assert out["ret"][0, 0] == 0.0
    assert out["entry_price"][0] == pytest.approx(10.1)


def test_first_passage_matches_bruteforce_oracle():
    time_ns, bid, ask = random_ticks(11)
    rng = np.random.default_rng(11)
    entries = np.sort(rng.choice(np.arange(1, len(time_ns) - 500), 40, replace=False))
    entries = entries.astype("int64")
    dirs = rng.choice(np.array([-1, 1], dtype="int8"), 40)
    targets = rng.uniform(0.2, 1.5, 40)
    stops = rng.uniform(0.2, 1.5, 40)
    horizon = np.int64(900 * 10**9)

    out = first_passage(time_ns, bid, ask, entries, dirs, targets, stops, horizon)

    outcomes = set()
    for k in range(40):
        oc, pnl, et = oracle_first_passage(time_ns, bid, ask, entries[k], dirs[k],
                                           targets[k], stops[k], horizon)
        outcomes.add(oc)
        assert out["outcome"][k] == oc, k
        assert out["pnl"][k] == pytest.approx(pnl, abs=1e-12), k
        assert out["exit_time_ns"][k] == et, k
    assert len(outcomes) > 1


def test_gap_through_target_pays_actual_crossing_price():
    time_ns = np.arange(4, dtype="int64") * 10**9
    bid = np.array([100.0, 100.0, 103.0, 103.0])
    ask = bid + 0.1

    out = first_passage(time_ns, bid, ask, np.array([0], dtype="int64"),
                        np.array([1], dtype="int8"), np.array([1.0]),
                        np.array([1.0]), np.int64(10 * 10**9))

    assert out["outcome"][0] == 1
    assert out["pnl"][0] == pytest.approx(103.0 - 100.1)
    assert out["exit_time_ns"][0] == 2 * 10**9


def test_timeout_reports_closeout_value():
    time_ns = np.arange(4, dtype="int64") * 10**9
    bid = np.array([100.0, 100.2, 100.1, 105.0])
    ask = bid + 0.1

    out = first_passage(time_ns, bid, ask, np.array([0], dtype="int64"),
                        np.array([1], dtype="int8"), np.array([2.0]),
                        np.array([2.0]), np.int64(2 * 10**9))

    assert out["outcome"][0] == 0
    assert out["pnl"][0] == pytest.approx(100.1 - 100.1)
    assert out["exit_time_ns"][0] == 2 * 10**9


def test_stop_first_for_sell():
    time_ns = np.arange(3, dtype="int64") * 10**9
    bid = np.array([100.0, 101.0, 101.0])
    ask = bid + 0.1

    out = first_passage(time_ns, bid, ask, np.array([0], dtype="int64"),
                        np.array([-1], dtype="int8"), np.array([0.5]),
                        np.array([0.5]), np.int64(10 * 10**9))

    assert out["outcome"][0] == -1
    assert out["pnl"][0] == pytest.approx(100.0 - 101.1)
    assert out["exit_time_ns"][0] == 10**9
