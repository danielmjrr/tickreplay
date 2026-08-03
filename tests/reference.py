"""Golden-reference replay engine: a deliberately plain tick-by-tick Python loop.

This is the correctness anchor the numba kernel is judged against in
test_engine.py. Slow on purpose — clarity over speed. Both implementations were
written independently against the same semantics; the tests require them to agree
trade-for-trade, byte-for-byte.
"""
import numpy as np

_COLUMNS = ("dir", "entry_idx", "exit_idx", "entry_time_ns", "exit_time_ns",
            "entry_price", "exit_price", "pnl_gross", "pnl_net", "mfe", "mae",
            "exit_reason")
_DTYPES = {"dir": "int8", "entry_idx": "int64", "exit_idx": "int64",
           "entry_time_ns": "int64", "exit_time_ns": "int64",
           "entry_price": "float64", "exit_price": "float64",
           "pnl_gross": "float64", "pnl_net": "float64",
           "mfe": "float64", "mae": "float64", "exit_reason": "int8"}


def replay(time_ns, bid, ask, bar_tick_last, bar_signal, sl_dist, tp_dist,
           allow_buy=True, allow_sell=True, max_hold_ns=0, exit_on_opposite=False,
           cost=0.0):
    n_ticks = len(time_ns)

    # Scatter each bar's signal onto its entry tick for O(1) lookup in the loop.
    # A bar whose entry tick falls beyond the data is dropped (no-trade rule).
    entry_signal = np.zeros(n_ticks, dtype="int8")
    entry_sl = np.zeros(n_ticks, dtype="float64")
    entry_tp = np.zeros(n_ticks, dtype="float64")
    for b in range(len(bar_tick_last)):
        if bar_signal[b] == 0:
            continue
        i = bar_tick_last[b] + 1
        if i >= n_ticks:
            continue
        entry_signal[i] = bar_signal[b]
        entry_sl[i] = sl_dist[b]
        entry_tp[i] = tp_dist[b]

    rows = {col: [] for col in _COLUMNS}
    pos_dir = 0  # 0 = flat, +1 = long, -1 = short
    entry_idx = entry_time = entry_price = None
    locked_sl = locked_tp = 0.0
    mfe = mae = 0.0

    def open_trade(i, direction, sl, tp):
        nonlocal pos_dir, entry_idx, entry_time, entry_price, locked_sl, locked_tp, mfe, mae
        pos_dir = direction
        entry_idx = i
        entry_time = time_ns[i]
        entry_price = ask[i] if direction == 1 else bid[i]
        locked_sl, locked_tp = sl, tp
        mfe, mae = 0.0, 0.0

    def close_trade(i, reason):
        nonlocal pos_dir
        exit_price = bid[i] if pos_dir == 1 else ask[i]
        pnl = (exit_price - entry_price) if pos_dir == 1 else (entry_price - exit_price)
        rows["dir"].append(pos_dir)
        rows["entry_idx"].append(entry_idx)
        rows["exit_idx"].append(i)
        rows["entry_time_ns"].append(entry_time)
        rows["exit_time_ns"].append(time_ns[i])
        rows["entry_price"].append(entry_price)
        rows["exit_price"].append(exit_price)
        rows["pnl_gross"].append(pnl)
        rows["pnl_net"].append(pnl - cost)
        rows["mfe"].append(mfe)
        rows["mae"].append(mae)
        rows["exit_reason"].append(reason)
        pos_dir = 0

    for i in range(n_ticks):
        if pos_dir != 0:
            pnl_raw = (bid[i] - entry_price) if pos_dir == 1 else (entry_price - ask[i])
            mfe = max(mfe, pnl_raw)
            mae = max(mae, -pnl_raw)

            exited = True
            if locked_sl > 0 and pnl_raw <= -locked_sl:
                close_trade(i, 1)
            elif locked_tp > 0 and pnl_raw >= locked_tp:
                close_trade(i, 2)
            elif max_hold_ns > 0 and time_ns[i] >= entry_time + max_hold_ns:
                close_trade(i, 3)
            elif exit_on_opposite and entry_signal[i] != 0 and entry_signal[i] == -pos_dir:
                close_trade(i, 4)
            else:
                exited = False

            if exited:
                continue  # the exit tick never also opens a trade -- signals don't queue

        # an entry on the final tick (nothing after it to evaluate) is no trade;
        # the scattered signal still counts for opposite-exit detection above
        if pos_dir == 0 and i < n_ticks - 1:
            sig = entry_signal[i]
            if sig == 1 and allow_buy:
                open_trade(i, 1, entry_sl[i], entry_tp[i])
            elif sig == -1 and allow_sell:
                open_trade(i, -1, entry_sl[i], entry_tp[i])

    if pos_dir != 0:
        close_trade(n_ticks - 1, 5)

    return {col: np.array(rows[col], dtype=_DTYPES[col]) for col in _COLUMNS}
