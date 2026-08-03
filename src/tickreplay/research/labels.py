"""Per-signal tick-level labels for research (not trading).

Unlike the replay engine, every signal gets labeled — overlap allowed — because
these are measurement tools: excursion profiles and barrier outcomes feed statistics,
not a trade ledger. The marking convention matches the engine: longs to bid, shorts
to ask, entry at the first tick after the signal bar.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _excursions(time_ns, bid, ask, entry_ticks, dirs, checkpoints_ns,
                mfe, mae, ret, entry_price):
    n_ticks = time_ns.shape[0]
    n_cp = checkpoints_ns.shape[0]
    for k in prange(entry_ticks.shape[0]):
        e = entry_ticks[k]
        d = dirs[k]
        ep = ask[e] if d == 1 else bid[e]
        entry_price[k] = ep
        t0 = time_ns[e]
        f = 0.0
        a = 0.0
        pnl = 0.0
        j = e + 1
        for ci in range(n_cp):
            limit = t0 + checkpoints_ns[ci]
            while j < n_ticks and time_ns[j] <= limit:
                pnl = bid[j] - ep if d == 1 else ep - ask[j]
                if pnl > f:
                    f = pnl
                if -pnl > a:
                    a = -pnl
                j += 1
            mfe[k, ci] = f
            mae[k, ci] = a
            ret[k, ci] = pnl  # signed value at the last tick <= horizon


def excursions(time_ns, bid, ask, entry_ticks, dirs, checkpoints_ns):
    """Running MFE/MAE and close-out return at each horizon checkpoint (ascending).
    Returns dict of (n_signals, n_checkpoints) arrays plus entry prices."""
    n, m = len(entry_ticks), len(checkpoints_ns)
    mfe = np.zeros((n, m))
    mae = np.zeros((n, m))
    ret = np.zeros((n, m))
    entry_price = np.zeros(n)
    _excursions(np.ascontiguousarray(time_ns, np.int64),
                np.ascontiguousarray(bid, np.float64),
                np.ascontiguousarray(ask, np.float64),
                np.ascontiguousarray(entry_ticks, np.int64),
                np.ascontiguousarray(dirs, np.int8),
                np.ascontiguousarray(checkpoints_ns, np.int64),
                mfe, mae, ret, entry_price)
    return {"mfe": mfe, "mae": mae, "ret": ret, "entry_price": entry_price}


@njit(cache=True, parallel=True)
def _first_passage(time_ns, bid, ask, entry_ticks, dirs, targets, stops, horizon_ns,
                   outcome, pnl_out, exit_time):
    n_ticks = time_ns.shape[0]
    for k in prange(entry_ticks.shape[0]):
        e = entry_ticks[k]
        d = dirs[k]
        ep = ask[e] if d == 1 else bid[e]
        limit = time_ns[e] + horizon_ns
        pnl = 0.0
        oc = 0
        et = min(limit, time_ns[n_ticks - 1])
        j = e + 1
        while j < n_ticks and time_ns[j] <= limit:
            pnl = bid[j] - ep if d == 1 else ep - ask[j]
            if pnl >= targets[k]:
                oc, et = 1, time_ns[j]
                break
            if -pnl >= stops[k]:
                oc, et = -1, time_ns[j]
                break
            j += 1
        outcome[k] = oc
        pnl_out[k] = pnl
        exit_time[k] = et


def first_passage(time_ns, bid, ask, entry_ticks, dirs, targets, stops, horizon_ns):
    """Which barrier is touched first within the horizon: +1 target, -1 stop,
    0 timeout. On ticks the ordering is exact (no OHLC path ambiguity), and a gap
    through a barrier pays the real crossing price, not the barrier level."""
    n = len(entry_ticks)
    outcome = np.zeros(n, np.int8)
    pnl = np.zeros(n)
    exit_time = np.zeros(n, np.int64)
    _first_passage(np.ascontiguousarray(time_ns, np.int64),
                   np.ascontiguousarray(bid, np.float64),
                   np.ascontiguousarray(ask, np.float64),
                   np.ascontiguousarray(entry_ticks, np.int64),
                   np.ascontiguousarray(dirs, np.int8),
                   np.ascontiguousarray(targets, np.float64),
                   np.ascontiguousarray(stops, np.float64),
                   np.int64(horizon_ns), outcome, pnl, exit_time)
    return {"outcome": outcome, "pnl": pnl, "exit_time_ns": exit_time}
