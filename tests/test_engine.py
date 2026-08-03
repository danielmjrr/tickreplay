"""The replay kernel must agree with the golden reference, trade for trade."""
import numpy as np
import pandas as pd
import pytest

import reference
import tickreplay as tr

LEDGER = ("dir", "entry_idx", "exit_idx", "entry_time_ns", "exit_time_ns",
          "entry_price", "exit_price", "pnl_gross", "pnl_net", "mfe", "mae",
          "exit_reason")

CONFIGS = {
    "default": dict(),
    "max_hold": dict(max_hold=120),
    "exit_on_opposite": dict(exit_on_opposite=True),
    "combined_with_cost": dict(max_hold=300, exit_on_opposite=True, cost=0.09),
    "buy_only": dict(allow_sell=False),
    "sell_only": dict(allow_buy=False, max_hold=60),
}


def synthetic(seed, n_ticks=30_000, n_bars=500):
    """Irregular ticks, sparse signals, per-bar stops with some disabled (0)."""
    rng = np.random.default_rng(seed)
    time_ns = np.cumsum(rng.integers(10**8, 3 * 10**9, n_ticks)).astype("int64")
    bid = 4000 + np.cumsum(rng.normal(0, 0.05, n_ticks))
    ask = bid + rng.uniform(0.05, 0.2, n_ticks)
    tick_last = np.sort(rng.choice(np.arange(1, n_ticks - 1), n_bars, replace=False))
    signal = np.zeros(n_bars, dtype="int8")
    hot = rng.choice(n_bars, n_bars // 10, replace=False)
    signal[hot] = rng.choice(np.array([-1, 1], dtype="int8"), hot.size)
    sl = rng.uniform(0.2, 0.8, n_bars)
    tp = rng.uniform(0.2, 1.0, n_bars)
    sl[rng.random(n_bars) < 0.3] = 0.0
    tp[rng.random(n_bars) < 0.3] = 0.0
    # a traded bar with no stop can hold to the end of data under the stopless default
    # config and starve every later signal, so keep one side live on those bars
    sl[(signal != 0) & (sl == 0)] = 0.5
    store = tr.TickStore(time_ns, bid, ask)
    bars = pd.DataFrame({"tick_last": tick_last.astype("int64")})
    return store, bars, signal, sl, tp


def five_exit_reasons():
    """Hand-built ticks that fire each exit reason exactly once, in order.

    Ticks are 1s apart until index 104, then 0.1s apart so the final trade runs out
    of data before it runs out of max_hold.
    """
    n = 120
    time_ns = np.arange(n, dtype="int64") * 10**9
    time_ns[105:] = time_ns[104] + np.arange(1, n - 104, dtype="int64") * 10**8
    bid = np.full(n, 100.0)
    bid[7] = 99.4     # stop touch for the trade entered at tick 5
    bid[27] = 100.8   # target touch for the trade entered at tick 25
    store = tr.TickStore(time_ns, bid, bid + 0.1)
    bars = pd.DataFrame({"tick_last": np.array([4, 24, 44, 64, 68, 104], dtype="int64")})
    signal = np.array([1, 1, 1, 1, -1, 1], dtype="int8")
    sl = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    tp = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    return store, bars, signal, sl, tp


def run_reference(store, bars, signal, config):
    return reference.replay(
        np.asarray(store.time_ns), np.asarray(store.bid), np.asarray(store.ask),
        bars["tick_last"].to_numpy(), signal,
        np.broadcast_to(config.sl, len(bars)), np.broadcast_to(config.tp, len(bars)),
        allow_buy=config.allow_buy, allow_sell=config.allow_sell,
        max_hold_ns=config.max_hold_ns(), exit_on_opposite=config.exit_on_opposite,
        cost=config.cost)


def assert_ledgers_equal(a, b):
    assert len(a["dir"]) == len(b["dir"]), "trade count"
    for col in LEDGER:
        np.testing.assert_array_equal(a[col], b[col], err_msg=col)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("overrides", CONFIGS.values(), ids=CONFIGS)
def test_fast_engine_matches_golden_reference(seed, overrides):
    store, bars, signal, sl, tp = synthetic(seed)
    config = tr.Config(sl=sl, tp=tp, **overrides)
    fast = tr.replay(store, bars, signal, config).arrays
    assert len(fast["dir"]) > 0, "fixture produced no trades to compare"
    assert_ledgers_equal(fast, run_reference(store, bars, signal, config))


def test_golden_matrix_is_not_vacuous():
    """Guards the test above: the fixtures must trade a lot and hit every exit path."""
    reasons, trades = set(), 0
    for overrides in CONFIGS.values():
        for seed in range(5):
            store, bars, signal, sl, tp = synthetic(seed)
            arrays = tr.replay(store, bars, signal,
                               tr.Config(sl=sl, tp=tp, **overrides)).arrays
            reasons |= set(arrays["exit_reason"].tolist())
            trades += len(arrays["dir"])
    assert trades > 500
    assert reasons == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("side, overrides", [(1, CONFIGS["buy_only"]),
                                             (-1, CONFIGS["sell_only"])])
def test_direction_filters_take_only_their_own_side(side, overrides):
    store, bars, signal, sl, tp = synthetic(0)
    arrays = tr.replay(store, bars, signal,
                       tr.Config(sl=sl, tp=tp, **overrides)).arrays
    assert len(arrays["dir"]) > 3
    assert set(arrays["dir"].tolist()) == {side}


def test_fast_engine_matches_golden_reference_on_every_exit_reason():
    store, bars, signal, sl, tp = five_exit_reasons()
    config = tr.Config(sl=sl, tp=tp, max_hold=12, exit_on_opposite=True)
    fast = tr.replay(store, bars, signal, config).arrays
    assert list(fast["exit_reason"]) == [1, 2, 3, 4, 5]
    assert_ledgers_equal(fast, run_reference(store, bars, signal, config))


def test_replay_is_deterministic():
    store, bars, signal, sl, tp = synthetic(9)
    config = tr.Config(sl=sl, tp=tp, max_hold=300, exit_on_opposite=True, cost=0.09)
    first = tr.replay(store, bars, signal, config).arrays
    second = tr.replay(store, bars, signal, config).arrays
    assert len(first["dir"]) > 3
    assert_ledgers_equal(first, second)


def test_trades_closed_before_a_cut_do_not_depend_on_later_data():
    store, bars, signal, sl, tp = synthetic(3)
    overrides = dict(max_hold=100, exit_on_opposite=True, cost=0.09)
    full = tr.replay(store, bars, signal, tr.Config(sl=sl, tp=tp, **overrides)).arrays

    cut = len(store) // 2
    n_bars = int(np.searchsorted(bars["tick_last"].to_numpy(), cut - 1))
    truncated = tr.replay(
        tr.TickStore(store.time_ns[:cut], store.bid[:cut], store.ask[:cut]),
        bars.iloc[:n_bars], signal[:n_bars],
        tr.Config(sl=sl[:n_bars], tp=tp[:n_bars], **overrides)).arrays

    closed = full["exit_idx"] < cut - 1
    assert closed.sum() > 5
    for col in LEDGER:
        np.testing.assert_array_equal(truncated[col][:closed.sum()], full[col][closed],
                                      err_msg=col)


def test_scalar_stops_broadcast_to_every_bar():
    store, bars, signal, _, _ = synthetic(7)
    n = len(bars)
    scalar = tr.replay(store, bars, signal, tr.Config(sl=1.5, tp=2.5)).arrays
    per_bar = tr.replay(store, bars, signal,
                        tr.Config(sl=np.full(n, 1.5), tp=np.full(n, 2.5))).arrays
    assert len(scalar["dir"]) > 3
    assert_ledgers_equal(scalar, per_bar)


def test_signal_length_mismatch_raises():
    store, bars, signal, _, _ = synthetic(1)
    with pytest.raises(ValueError, match=r"499 values for 500 bars"):
        tr.replay(store, bars, signal[:-1])


def test_non_increasing_tick_last_raises():
    # duplicate tick_last would let two bars share an entry tick, where the fast and
    # reference engines resolve opposite-signal exits differently -- reject it outright
    store, bars, signal, _, _ = synthetic(1)
    bad = bars.copy()
    bad.loc[bad.index[10], "tick_last"] = bad["tick_last"].iloc[9]
    with pytest.raises(ValueError, match="strictly increasing"):
        tr.replay(store, bad, signal)


def test_trades_frame_exposes_datetimes_and_named_exit_reasons():
    store, bars, signal, sl, tp = five_exit_reasons()
    result = tr.replay(store, bars, signal,
                       tr.Config(sl=sl, tp=tp, max_hold=12, exit_on_opposite=True))
    df = result.trades

    assert len(df) == len(result) == 5
    assert df["entry_time"].dtype == np.dtype("datetime64[ns]")
    assert df["exit_time"].dtype == np.dtype("datetime64[ns]")
    assert list(df["exit_reason"]) == ["sl", "tp", "time", "opposite", "eod"]
    assert not {"entry_time_ns", "exit_time_ns"} & set(df.columns)
    np.testing.assert_array_equal(df["entry_time"].astype("int64").to_numpy(),
                                  result.arrays["entry_time_ns"])
    np.testing.assert_array_equal(df["exit_time"].astype("int64").to_numpy(),
                                  result.arrays["exit_time_ns"])
    assert df["pnl_gross"].to_numpy() == pytest.approx([-0.7, 0.7, -0.1, -0.1, -0.1])


def test_summary_agrees_with_the_ledger():
    store, bars, signal, sl, tp = synthetic(2)
    result = tr.replay(store, bars, signal,
                       tr.Config(sl=sl, tp=tp, max_hold=300, cost=0.09))
    pnl = result.arrays["pnl_net"]
    summary = result.summary()

    assert set(summary) == {"n_trades", "total_net", "mean_net", "hit_rate",
                            "max_drawdown"}
    assert summary["n_trades"] == len(pnl) == len(result)
    assert summary["total_net"] == pytest.approx(pnl.sum())
    assert summary["mean_net"] == pytest.approx(pnl.mean())
    assert summary["hit_rate"] == pytest.approx((pnl > 0).sum() / len(pnl))
    assert 0 < summary["hit_rate"] < 1

    equity = peak = drawdown = 0.0
    for i, x in enumerate(pnl):
        equity += x
        peak = equity if i == 0 else max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    assert summary["max_drawdown"] == pytest.approx(drawdown)


def test_max_hold_accepts_sub_second_and_offset_string_forms():
    n = 20
    price = np.full(n, 100.0)
    store = tr.TickStore(np.arange(n, dtype="int64") * 10**8, price, price + 0.1)
    bars = pd.DataFrame({"tick_last": np.array([4], dtype="int64")})
    signal = np.array([1], dtype="int8")  # enters at tick 5, i.e. t = 0.5s

    for max_hold in (0.25, "250ms"):
        result = tr.replay(store, bars, signal, tr.Config(max_hold=max_hold))
        assert result.arrays["exit_idx"].tolist() == [8], max_hold
        assert list(result.trades["exit_reason"]) == ["time"], max_hold


def test_signal_entering_on_the_last_tick_produces_no_trade():
    n = 20
    price = np.full(n, 100.0)
    store = tr.TickStore(np.arange(n, dtype="int64") * 10**9, price, price + 0.1)
    signal = np.array([1], dtype="int8")

    entering_on_last = pd.DataFrame({"tick_last": np.array([n - 2], dtype="int64")})
    entering_one_earlier = pd.DataFrame({"tick_last": np.array([n - 3], dtype="int64")})

    assert len(tr.replay(store, entering_on_last, signal)) == 0
    assert len(tr.replay(store, entering_one_earlier, signal)) == 1


def test_breakeven_trade_is_not_counted_as_a_win():
    n = 30
    price = np.full(n, 100.0)
    store = tr.TickStore(np.arange(n, dtype="int64") * 10**9, price, price)
    bars = pd.DataFrame({"tick_last": np.array([4], dtype="int64")})

    result = tr.replay(store, bars, np.array([1], dtype="int8"), tr.Config(max_hold=10))

    assert result.arrays["pnl_net"].tolist() == [0.0]
    assert result.summary()["hit_rate"] == 0.0


def test_summary_of_an_empty_ledger():
    store, bars, _, _, _ = synthetic(4)
    result = tr.replay(store, bars, np.zeros(len(bars), dtype="int8"))
    assert len(result) == 0
    assert result.summary() == {"n_trades": 0, "total_net": 0.0, "mean_net": 0.0,
                                "hit_rate": 0.0, "max_drawdown": 0.0}
