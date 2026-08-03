"""TickStore round-trips and bar derivation against a brute-force pandas oracle."""
import json

import numpy as np
import pandas as pd
import pytest

from tickreplay.store import TIMEFRAMES, TickStore, derive_bars


def _random_ticks(n=4000, seed=7, start="2024-01-02T00:00:00", max_gap_s=600):
    """Irregularly spaced ticks: gaps up to max_gap_s leave many buckets empty."""
    rng = np.random.default_rng(seed)
    t0 = np.datetime64(start, "ns").astype("int64")
    time_ns = (t0 + np.cumsum(rng.integers(1, max_gap_s * 10**9, size=n))).astype("int64")
    bid = 2000.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    ask = bid + rng.uniform(0.01, 0.05, n)
    return time_ns, bid, ask


def _oracle_bars(time_ns, bid, ask, timeframe):
    tf_ns = TIMEFRAMES[timeframe] * 10**9
    df = pd.DataFrame({"bucket": time_ns // tf_ns, "bid": bid, "spread": ask - bid})
    out = df.groupby("bucket", sort=True).agg(
        open=("bid", "first"), high=("bid", "max"), low=("bid", "min"),
        close=("bid", "last"), spread_mean=("spread", "mean"), tick_count=("bid", "size"))
    out.insert(0, "time", pd.to_datetime(out.index.to_numpy() * tf_ns, unit="ns"))
    return out.reset_index(drop=True)


def _month_frame(start, n, seed, max_gap_s=3600):
    rng = np.random.default_rng(seed)
    t0 = np.datetime64(start, "ns").astype("int64")
    times = t0 + np.cumsum(rng.integers(1, max_gap_s * 10**9, size=n))
    bid = 2000.0 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"time": pd.to_datetime(times, unit="ns"),
                         "bid": bid, "ask": bid + 0.03})


@pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
def test_derive_bars_matches_pandas_oracle(timeframe):
    time_ns, bid, ask = _random_ticks()
    bars = derive_bars(time_ns, bid, ask, timeframe)
    oracle = _oracle_bars(time_ns, bid, ask, timeframe)

    assert len(oracle) > 1
    assert len(bars) == len(oracle)
    assert np.array_equal(bars["time"].to_numpy(), oracle["time"].to_numpy())
    for col in ("open", "high", "low", "close"):
        assert np.array_equal(bars[col].to_numpy(), oracle[col].to_numpy())
    np.testing.assert_allclose(bars["spread_mean"].to_numpy(),
                               oracle["spread_mean"].to_numpy(), rtol=1e-12)
    assert np.array_equal(bars["tick_count"].to_numpy(), oracle["tick_count"].to_numpy())


def test_derive_bars_emits_no_bar_for_empty_buckets():
    time_ns, bid, ask = _random_ticks()
    tf_ns = TIMEFRAMES["M1"] * 10**9
    bars = derive_bars(time_ns, bid, ask, "M1")

    spanned = int(time_ns[-1] // tf_ns - time_ns[0] // tf_ns) + 1
    assert len(bars) < spanned
    occupied = np.unique(time_ns // tf_ns)
    assert np.array_equal(bars["time"].to_numpy().astype("int64") // tf_ns, occupied)


def test_bar_times_are_epoch_aligned():
    time_ns, bid, ask = _random_ticks()
    for timeframe, seconds in TIMEFRAMES.items():
        times = derive_bars(time_ns, bid, ask, timeframe)["time"]
        assert (times.to_numpy().astype("int64") % (seconds * 10**9) == 0).all()

    h4 = derive_bars(time_ns, bid, ask, "H4")["time"].dt
    assert set(h4.hour.unique()) == {0, 4, 8, 12, 16, 20}
    assert (h4.minute == 0).all()
    assert (h4.second == 0).all()

    d1 = derive_bars(time_ns, bid, ask, "D1")["time"]
    assert len(d1) > 1
    assert (d1 == d1.dt.normalize()).all()


@pytest.mark.parametrize("timeframe", ["M1", "M15", "H4"])
def test_tick_spans_partition_every_tick(timeframe):
    time_ns, bid, ask = _random_ticks()
    bars = derive_bars(time_ns, bid, ask, timeframe)
    first = bars["tick_first"].to_numpy()
    last = bars["tick_last"].to_numpy()
    count = bars["tick_count"].to_numpy()

    assert first[0] == 0
    assert last[-1] == len(time_ns) - 1
    assert (last >= first).all()
    assert np.array_equal(first[1:], last[:-1] + 1)
    assert np.array_equal(count, last - first + 1)
    assert count.sum() == len(time_ns)

    bucket = time_ns // (TIMEFRAMES[timeframe] * 10**9)
    assert np.array_equal(bucket[first], bucket[last])
    assert np.array_equal(bars["time"].to_numpy().astype("int64"),
                          bucket[first] * TIMEFRAMES[timeframe] * 10**9)


def test_write_open_roundtrip(tmp_path):
    time_ns, bid, ask = _random_ticks(n=500, seed=1)
    store_dir = tmp_path / "store"
    written = TickStore.write(store_dir, time_ns, bid, ask, {"symbol": "XAUUSD"})
    expected_range = [str(np.datetime64(int(time_ns[0]), "ns")),
                      str(np.datetime64(int(time_ns[-1]), "ns"))]

    for store in (written, TickStore.open(store_dir)):
        assert len(store) == len(time_ns)
        assert np.array_equal(np.asarray(store.time_ns), time_ns)
        assert np.array_equal(np.asarray(store.bid), bid)
        assert np.array_equal(np.asarray(store.ask), ask)
        assert store.manifest["n_ticks"] == len(time_ns)
        assert store.manifest["time_range"] == expected_range
        assert store.manifest["symbol"] == "XAUUSD"

    assert json.loads((store_dir / "manifest.json").read_text()) == written.manifest


def test_write_rejects_non_monotonic_time(tmp_path):
    backwards = np.array([10, 30, 20], dtype="int64") * 10**9
    with pytest.raises(ValueError, match="non-monotonic"):
        TickStore.write(tmp_path / "bad", backwards, [1.0, 2.0, 3.0], [1.1, 2.1, 3.1])

    repeated = np.array([10, 20, 20, 30], dtype="int64") * 10**9
    store = TickStore.write(tmp_path / "ok", repeated, [1.0] * 4, [1.1] * 4)
    assert len(store) == 4


def test_build_concatenates_monthly_parquet_files(tmp_path):
    months = {"2024-01": _month_frame("2024-01-02T00:00:00", 50, 0),
              "2024-02": _month_frame("2024-02-01T00:00:00", 40, 1),
              "2024-03": _month_frame("2024-03-01T00:00:00", 30, 2)}
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    for month, frame in months.items():
        frame.to_parquet(parquet_dir / f"ticks_{month}.parquet", index=False)

    store = TickStore.build(parquet_dir, tmp_path / "store")
    expected = pd.concat(months.values(), ignore_index=True)

    assert len(store) == len(expected) == 120
    assert np.array_equal(np.asarray(store.time_ns),
                          expected["time"].to_numpy().astype("datetime64[ns]").astype("int64"))
    assert np.array_equal(np.asarray(store.bid), expected["bid"].to_numpy())
    assert np.array_equal(np.asarray(store.ask), expected["ask"].to_numpy())
    assert store.manifest["n_ticks"] == 120
    assert store.manifest["sources"] == ["ticks_2024-01.parquet", "ticks_2024-02.parquet",
                                         "ticks_2024-03.parquet"]
    assert store.manifest["time_range"][0].startswith("2024-01")
    assert store.manifest["time_range"][1].startswith("2024-03")
    assert TickStore.open(tmp_path / "store").manifest == store.manifest


def test_build_rejects_non_monotonic_time_within_a_file(tmp_path):
    frame = _month_frame("2024-01-02T00:00:00", 20, 0)
    frame.loc[10, "time"] = frame.loc[2, "time"]
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    frame.to_parquet(parquet_dir / "ticks_2024-01.parquet", index=False)

    with pytest.raises(ValueError, match="non-monotonic"):
        TickStore.build(parquet_dir, tmp_path / "store")


def test_build_rejects_time_going_backwards_across_files(tmp_path):
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    _month_frame("2024-02-01T00:00:00", 20, 0).to_parquet(
        parquet_dir / "ticks_2024-01.parquet", index=False)
    _month_frame("2024-01-02T00:00:00", 20, 1).to_parquet(
        parquet_dir / "ticks_2024-02.parquet", index=False)

    with pytest.raises(ValueError, match="non-monotonic"):
        TickStore.build(parquet_dir, tmp_path / "store")


def test_build_requires_at_least_one_parquet_file(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no ticks_"):
        TickStore.build(tmp_path / "empty", tmp_path / "store")


def test_bars_are_cached_per_timeframe(tmp_path):
    time_ns, bid, ask = _random_ticks(n=800, seed=3)
    store = TickStore.write(tmp_path / "store", time_ns, bid, ask)

    m5 = store.bars("M5")
    assert store.bars("M5") is m5
    assert store.bars("M15") is not m5
    pd.testing.assert_frame_equal(m5, derive_bars(time_ns, bid, ask, "M5"))
