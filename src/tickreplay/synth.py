"""Synthetic tick data — lets examples, tests, and benchmarks run without broker data.

Not a market model: a regime-switching random walk with irregular tick arrivals,
session-shaped spread, and weekend gaps. Realistic enough to exercise every engine
code path (stops, targets, time exits, quiet periods, spread costs).
"""
import numpy as np


def generate_ticks(days=5, ticks_per_day=50_000, start="2024-01-01", price=2000.0,
                   tick_vol=0.02, spread=0.10, seed=0):
    """Returns (time_ns, bid, ask). Weekdays only; ~22h of trading per day."""
    rng = np.random.default_rng(seed)
    day0 = np.datetime64(start).astype("datetime64[D]")
    times, mids, spreads = [], [], []
    mid = float(price)
    for d in range(days):
        day = day0 + d
        if np.datetime64(day).astype("datetime64[D]").item().weekday() >= 5:
            continue
        # irregular arrivals over 22 hours
        gaps = rng.exponential(22 * 3600 / ticks_per_day, ticks_per_day)
        t = np.cumsum(gaps)
        t = t[t < 22 * 3600]
        # two-state volatility regime (calm/wild), persistent at tick scale
        switch = rng.random(len(t)) < 0.0005
        regime = np.bitwise_xor.accumulate(switch)
        vol = np.where(regime, 4.0 * tick_vol, tick_vol)
        steps = rng.normal(0.0, 1.0, len(t)) * vol
        path = mid + np.cumsum(steps)
        mid = float(path[-1])
        # spread widens outside core hours and in the wild regime
        hour = t / 3600
        session = 1.0 + 0.8 * ((hour < 1) | (hour > 21)) + 0.5 * regime
        spreads.append(spread * session * rng.uniform(0.8, 1.2, len(t)))
        times.append(day.astype("datetime64[ns]").astype("int64")
                     + (t * 1e9).astype("int64"))
        mids.append(path)
    if not times:
        raise ValueError(f"no weekdays in the requested window ({start} + {days} days)")
    time_ns = np.concatenate(times)
    mid_all = np.concatenate(mids)
    half = np.concatenate(spreads) / 2
    return time_ns, mid_all - half, mid_all + half
