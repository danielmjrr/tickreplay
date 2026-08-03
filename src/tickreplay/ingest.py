"""MT5 export ingestion: tick CSV -> monthly parquet, plus reconstruction checks.

MT5 tick exports are tab-separated: <DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>.
Rows may carry only one quote side; the missing side is forward-filled (quote state),
matching how the terminal itself maintains the current bid/ask. MT5 bars are bid-based.
"""
import json
from pathlib import Path

import pandas as pd


def _parse_time(date_col, time_col):
    s = date_col + " " + time_col
    try:
        return pd.to_datetime(s, format="%Y.%m.%d %H:%M:%S.%f")
    except ValueError:
        return pd.to_datetime(s, format="mixed", dayfirst=False)


def read_mt5_ticks(src, chunksize=5_000_000):
    """Yield DataFrames [time, bid, ask] with quote state forward-filled across chunks.
    Rows before both sides are first known are dropped."""
    carry_bid = float("nan")
    carry_ask = float("nan")
    for chunk in pd.read_csv(src, sep="\t", chunksize=chunksize):
        chunk.columns = [c.strip("<>").lower() for c in chunk.columns]
        bid = pd.concat([pd.Series([carry_bid]), chunk["bid"].astype("float64")],
                        ignore_index=True).ffill().iloc[1:].reset_index(drop=True)
        ask = pd.concat([pd.Series([carry_ask]), chunk["ask"].astype("float64")],
                        ignore_index=True).ffill().iloc[1:].reset_index(drop=True)
        carry_bid = bid.iloc[-1]
        carry_ask = ask.iloc[-1]
        df = pd.DataFrame({
            "time": _parse_time(chunk["date"], chunk["time"]).reset_index(drop=True),
            "bid": bid,
            "ask": ask,
        }).dropna(subset=["bid", "ask"]).reset_index(drop=True)
        if len(df):
            yield df


def ticks_to_monthly_parquet(src, out_dir, chunksize=5_000_000):
    """Convert an MT5 tick CSV to per-month parquet files; returns a manifest.

    Assumes chronological input (MT5 exports are). A month is flushed once a later
    month appears, keeping peak memory at ~one month + one chunk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffers = {}
    months_written = {}
    t_min = t_max = None

    def flush(month):
        df = pd.concat(buffers.pop(month), ignore_index=True)
        path = out_dir / f"ticks_{month}.parquet"
        if path.exists():
            raise RuntimeError(f"non-chronological input: {month} seen again after flush")
        df.to_parquet(path, index=False)
        months_written[str(month)] = len(df)

    for df in read_mt5_ticks(src, chunksize=chunksize):
        t_min = df["time"].iloc[0] if t_min is None else t_min
        t_max = df["time"].iloc[-1]
        for month, part in df.groupby(df["time"].dt.to_period("M"), sort=True):
            buffers.setdefault(str(month), []).append(part.reset_index(drop=True))
        latest = max(buffers)
        for month in [m for m in sorted(buffers) if m != latest]:
            flush(month)
    for month in sorted(buffers):
        flush(month)

    manifest = {
        "source": str(src),
        "total_ticks": int(sum(months_written.values())),
        "months": months_written,
        "range": [str(t_min), str(t_max)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_mt5_bars(src):
    """Parse an MT5 bar CSV (any timeframe) to a time-indexed OHLC frame."""
    df = pd.read_csv(src, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%Y.%m.%d %H:%M:%S")
    else:  # daily and above: no <TIME> column
        df["time"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
    return df.set_index("time")[["open", "high", "low", "close", "tickvol", "spread"]]


def reconstruct_m1(ticks):
    """Bid-based M1 OHLC from a tick frame; minutes without ticks produce no bar."""
    g = ticks.set_index("time")["bid"].resample("1min")
    bars = g.agg(["first", "max", "min", "last"]).dropna()
    bars.columns = ["open", "high", "low", "close"]
    return bars


def validate_against_broker_bars(ticks, broker_bars):
    """Compare tick-reconstructed M1 OHLC against broker-exported bars on the overlap.

    Zero mismatches is the expected outcome for a clean export; anything else usually
    means the tick and bar files come from different terminals or accounts.
    """
    recon = reconstruct_m1(ticks)
    j = recon.join(broker_bars[["open", "high", "low", "close"]],
                   how="inner", lsuffix="_r", rsuffix="_b")
    mism = j[(j.open_r != j.open_b) | (j.high_r != j.high_b)
             | (j.low_r != j.low_b) | (j.close_r != j.close_b)]
    in_range = broker_bars.index[(broker_bars.index >= recon.index.min())
                                 & (broker_bars.index <= recon.index.max())]
    return {
        "compared": int(len(j)),
        "ohlc_mismatches": int(len(mism)),
        "mismatch_sample": [str(t) for t in mism.index[:20]],
        "broker_only": int(len(in_range.difference(recon.index))),
        "recon_only": int(len(recon.index.difference(broker_bars.index))),
    }
