"""Tick storage and bar derivation.

A tickstore is a directory of flat little-endian arrays (``time_ns.npy``, ``bid.npy``,
``ask.npy``) plus a manifest. Loading is zero-copy via ``np.memmap``, and the OS page
cache shares physical pages across processes — parallel workers cost no extra RAM
for data.

Bars are derived from ticks on demand. Each bar carries its tick index span
(``tick_first``/``tick_last``), which is both an O(1) bar-to-ticks mapping and the
no-look-ahead boundary: a bar's values exist only after its last tick has printed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TIMEFRAMES = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600,
              "H4": 14400, "D1": 86400}

_ARRAYS = {"time_ns": np.int64, "bid": np.float64, "ask": np.float64}


def derive_bars(time_ns, bid, ask, timeframe):
    """Bid-based OHLC bars with tick index spans, epoch-aligned like MT5
    (H4 bars at 00/04/08..., D1 at midnight). Empty buckets produce no bar, so bar
    times are not calendar-uniform: the row index is positional, and rolling windows
    silently span session and weekend gaps — same as bars in the MT5 terminal."""
    tf_ns = TIMEFRAMES[timeframe] * 10**9
    bucket = time_ns // tf_ns
    first = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    last = np.r_[first[1:] - 1, len(bucket) - 1]
    count = last - first + 1
    return pd.DataFrame({
        "time": pd.to_datetime(bucket[first] * tf_ns, unit="ns"),
        "open": bid[first],
        "high": np.maximum.reduceat(bid, first),
        "low": np.minimum.reduceat(bid, first),
        "close": bid[last],
        "spread_mean": np.add.reduceat(ask - bid, first) / count,
        "tick_first": first.astype("int64"),
        "tick_last": last.astype("int64"),
        "tick_count": count.astype("int64"),
    })


class TickStore:
    """Memory-mapped tick data for one symbol.

    Build once from parquet files or in-memory arrays, then ``open`` cheaply from
    any process. ``bars`` results are cached per timeframe.
    """

    def __init__(self, time_ns, bid, ask, manifest=None):
        self.time_ns = time_ns
        self.bid = bid
        self.ask = ask
        self.manifest = manifest or {}
        self._bars = {}

    def __len__(self):
        return len(self.time_ns)

    @classmethod
    def open(cls, store_dir):
        store_dir = Path(store_dir)
        arrays = {name: np.load(store_dir / f"{name}.npy", mmap_mode="r")
                  for name in _ARRAYS}
        manifest = json.loads((store_dir / "manifest.json").read_text())
        return cls(arrays["time_ns"], arrays["bid"], arrays["ask"], manifest)

    @classmethod
    def build(cls, parquet_dir, out_dir):
        """Concatenate ``ticks_*.parquet`` files (chronological order by filename)
        into a tickstore. Rejects non-monotonic timestamps."""
        files = sorted(Path(parquet_dir).glob("ticks_*.parquet"))
        if not files:
            raise ValueError(f"no ticks_*.parquet in {parquet_dir}")
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        maps = {name: np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+",
                                                dtype=dt, shape=(total,))
                for name, dt in _ARRAYS.items()}
        pos = 0
        last_t = None
        for f in files:
            tbl = pq.read_table(f, columns=["time", "bid", "ask"])
            t = tbl["time"].to_numpy().astype("datetime64[ns]").astype("int64")
            if (last_t is not None and len(t) and t[0] < last_t) or (np.diff(t) < 0).any():
                raise ValueError(f"non-monotonic time in {f.name}")
            maps["time_ns"][pos:pos + len(t)] = t
            maps["bid"][pos:pos + len(t)] = tbl["bid"].to_numpy()
            maps["ask"][pos:pos + len(t)] = tbl["ask"].to_numpy()
            last_t = t[-1] if len(t) else last_t
            pos += len(t)
        for m in maps.values():
            m.flush()
        manifest = {
            "n_ticks": int(total),
            "time_range": [str(np.datetime64(int(maps["time_ns"][0]), "ns")),
                           str(np.datetime64(int(maps["time_ns"][-1]), "ns"))],
            "sources": [f.name for f in files],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return cls.open(out_dir)

    @classmethod
    def write(cls, out_dir, time_ns, bid, ask, extra_manifest=None):
        """Write in-memory arrays as a tickstore (used for synthetic data and tests)."""
        time_ns = np.ascontiguousarray(time_ns, np.int64)
        if (np.diff(time_ns) < 0).any():
            raise ValueError("non-monotonic time")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "time_ns.npy", time_ns)
        np.save(out_dir / "bid.npy", np.ascontiguousarray(bid, np.float64))
        np.save(out_dir / "ask.npy", np.ascontiguousarray(ask, np.float64))
        manifest = {
            "n_ticks": int(len(time_ns)),
            "time_range": [str(np.datetime64(int(time_ns[0]), "ns")),
                           str(np.datetime64(int(time_ns[-1]), "ns"))],
            **(extra_manifest or {}),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return cls.open(out_dir)

    def bars(self, timeframe):
        if timeframe not in self._bars:
            self._bars[timeframe] = derive_bars(
                np.asarray(self.time_ns), np.asarray(self.bid),
                np.asarray(self.ask), timeframe)
        return self._bars[timeframe]
