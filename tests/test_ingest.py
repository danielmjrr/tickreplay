"""MT5 export ingestion: quote forward-fill, monthly parquet split, M1 reconstruction."""
import json

import numpy as np
import pandas as pd
import pytest

from tickreplay.ingest import (read_mt5_ticks, reconstruct_m1, ticks_to_monthly_parquet,
                               validate_against_broker_bars)

HEADER = "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>"

# (date, time, bid, ask) — "" means the row carries only the other quote side.
ONE_SIDED_ROWS = [
    ("2024.01.15", "10:30:00.000", "", ""),
    ("2024.01.15", "10:30:00.100", "2000.10", ""),
    ("2024.01.15", "10:30:00.200", "", "2000.30"),
    ("2024.01.15", "10:30:00.300", "", ""),
    ("2024.01.15", "10:30:00.400", "2000.50", ""),
    ("2024.01.15", "10:30:00.500", "", "2000.70"),
]


def _write_tick_csv(path, rows):
    lines = [HEADER] + [f"{d}\t{t}\t{bid}\t{ask}\t0\t0\t2" for d, t, bid, ask in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _day_rows(date, n, base):
    return [(date, f"10:{m:02d}:00.000", f"{base + m * 0.1:.2f}",
             f"{base + m * 0.1 + 0.03:.2f}") for m in range(n)]


def _ticks():
    return pd.DataFrame({
        "time": pd.to_datetime([
            "2024-01-15 10:30:00.100", "2024-01-15 10:30:20.000",
            "2024-01-15 10:30:40.000", "2024-01-15 10:30:59.900",
            "2024-01-15 10:31:05.000", "2024-01-15 10:31:50.000",
            "2024-01-15 10:33:10.000", "2024-01-15 10:34:00.000",
        ]),
        "bid": [2000.0, 2001.5, 1999.5, 2000.5, 2000.6, 2000.9, 2002.0, 2003.25],
        "ask": [2000.03, 2001.53, 1999.53, 2000.53, 2000.63, 2000.93, 2002.03, 2003.28],
    })


def _broker_bars():
    """Hand-computed M1 bars for _ticks(), plus a 10:32 bar the ticks do not cover."""
    return pd.DataFrame(
        {"open": [2000.00, 2000.6, 2001.0, 2002.0],
         "high": [2001.50, 2000.9, 2001.2, 2002.0],
         "low": [1999.50, 2000.6, 2000.8, 2002.0],
         "close": [2000.50, 2000.9, 2001.1, 2002.0],
         "tickvol": [4, 2, 3, 1],
         "spread": [3, 3, 3, 3]},
        index=pd.DatetimeIndex(["2024-01-15 10:30", "2024-01-15 10:31",
                                "2024-01-15 10:32", "2024-01-15 10:33"], name="time"))


def test_read_mt5_ticks_forward_fills_missing_quote_side(tmp_path):
    src = _write_tick_csv(tmp_path / "ticks.csv", ONE_SIDED_ROWS)
    df = pd.concat(read_mt5_ticks(src), ignore_index=True)

    assert list(df.columns) == ["time", "bid", "ask"]
    expected_times = pd.to_datetime([
        "2024-01-15 10:30:00.200", "2024-01-15 10:30:00.300",
        "2024-01-15 10:30:00.400", "2024-01-15 10:30:00.500"])
    assert np.array_equal(df["time"].to_numpy(), expected_times.to_numpy())
    assert df["bid"].tolist() == [2000.10, 2000.10, 2000.50, 2000.50]
    assert df["ask"].tolist() == [2000.30, 2000.30, 2000.30, 2000.70]


def test_read_mt5_ticks_carries_quote_state_across_chunk_boundaries(tmp_path):
    src = _write_tick_csv(tmp_path / "ticks.csv", ONE_SIDED_ROWS)
    whole = pd.concat(read_mt5_ticks(src), ignore_index=True)
    chunks = list(read_mt5_ticks(src, chunksize=3))

    assert len(chunks) == 2
    chunked = pd.concat(chunks, ignore_index=True)
    # The first row of chunk 2 is only keepable because both sides carried over.
    assert len(chunked) == 4
    assert chunked["bid"].iloc[1] == 2000.10
    assert chunked["ask"].iloc[1] == 2000.30
    pd.testing.assert_frame_equal(chunked, whole)


def test_ticks_to_monthly_parquet_splits_rows_by_month(tmp_path):
    rows = (_day_rows("2024.01.15", 5, 2000.0) + _day_rows("2024.01.20", 4, 2010.0)
            + _day_rows("2024.02.03", 6, 2020.0) + _day_rows("2024.03.02", 3, 2030.0))
    src = _write_tick_csv(tmp_path / "ticks.csv", rows)
    out_dir = tmp_path / "parquet"

    manifest = ticks_to_monthly_parquet(src, out_dir, chunksize=4)

    assert sorted(p.name for p in out_dir.glob("*.parquet")) == [
        "ticks_2024-01.parquet", "ticks_2024-02.parquet", "ticks_2024-03.parquet"]
    assert manifest["months"] == {"2024-01": 9, "2024-02": 6, "2024-03": 3}
    assert manifest["total_ticks"] == len(rows) == 18
    assert manifest["range"] == ["2024-01-15 10:00:00", "2024-03-02 10:02:00"]
    assert json.loads((out_dir / "manifest.json").read_text()) == manifest

    for month in manifest["months"]:
        part = pd.read_parquet(out_dir / f"ticks_{month}.parquet")
        assert (part["time"].dt.to_period("M").astype(str) == month).all()

    combined = pd.concat([pd.read_parquet(out_dir / f"ticks_{m}.parquet")
                          for m in sorted(manifest["months"])], ignore_index=True)
    pd.testing.assert_frame_equal(combined, pd.concat(read_mt5_ticks(src), ignore_index=True))


def test_ticks_to_monthly_parquet_rejects_non_chronological_input(tmp_path):
    rows = (_day_rows("2024.01.15", 4, 2000.0) + _day_rows("2024.02.03", 4, 2020.0)
            + _day_rows("2024.01.20", 4, 2010.0))
    src = _write_tick_csv(tmp_path / "ticks.csv", rows)

    with pytest.raises(RuntimeError, match="non-chronological"):
        ticks_to_monthly_parquet(src, tmp_path / "parquet", chunksize=4)


def test_reconstruct_m1_matches_hand_computed_bars():
    bars = reconstruct_m1(_ticks())

    assert list(bars.columns) == ["open", "high", "low", "close"]
    expected_index = pd.to_datetime(["2024-01-15 10:30", "2024-01-15 10:31",
                                     "2024-01-15 10:33", "2024-01-15 10:34"])
    assert np.array_equal(bars.index.to_numpy(), expected_index.to_numpy())
    assert np.array_equal(bars.to_numpy(), np.array([
        [2000.00, 2001.50, 1999.50, 2000.50],
        [2000.60, 2000.90, 2000.60, 2000.90],
        [2002.00, 2002.00, 2002.00, 2002.00],
        [2003.25, 2003.25, 2003.25, 2003.25]]))


def test_validate_against_broker_bars_reports_no_mismatch_for_clean_export():
    result = validate_against_broker_bars(_ticks(), _broker_bars())

    assert result["compared"] == 3
    assert result["ohlc_mismatches"] == 0
    assert result["mismatch_sample"] == []
    assert result["broker_only"] == 1
    assert result["recon_only"] == 1


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_validate_against_broker_bars_flags_a_corrupted_bar(column):
    broker = _broker_bars()
    broker.loc[pd.Timestamp("2024-01-15 10:31"), column] = 9999.0

    result = validate_against_broker_bars(_ticks(), broker)

    assert result["compared"] == 3
    assert result["ohlc_mismatches"] == 1
    assert result["mismatch_sample"] == ["2024-01-15 10:31:00"]
