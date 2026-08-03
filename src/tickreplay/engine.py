"""Single-position tick replay.

Semantics, chosen to match how a live EA on a netting account behaves:

- A nonzero signal on bar ``b`` enters at the first tick after the bar closes
  (``tick_last + 1``) — buys at ask, sells at bid. No same-bar fills, no look-ahead.
- One position at a time. Signals arriving while a position is open are skipped, so
  results measure the *tradable* expectancy of the strategy, not an overlapped ideal.
- Exit checks per tick, in priority order: stop loss, take profit, max hold time,
  opposite signal. Anything still open at the last tick closes there (``eod``).
- Positions are marked to the liquidation side: longs to bid, shorts to ask. Stops
  and targets are distances from the entry price in price units and may vary per bar
  (e.g. ATR-scaled).
- A signal on the exit tick itself is ignored; the next entry needs a later tick.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

EXIT_REASONS = {1: "sl", 2: "tp", 3: "time", 4: "opposite", 5: "eod"}


@dataclass
class Config:
    """Exit rules and costs for one replay.

    ``sl``/``tp``: distance from entry in price units — a scalar for all bars or an
    array per bar; 0 disables. ``max_hold``: pandas offset string ("4h") or seconds;
    0 disables. ``cost``: round-trip cost added on top of the spread, in price units
    (spread itself is already paid via bid/ask fills).
    """
    sl: float | np.ndarray = 0.0
    tp: float | np.ndarray = 0.0
    max_hold: str | float = 0
    exit_on_opposite: bool = False
    allow_buy: bool = True
    allow_sell: bool = True
    cost: float = 0.0

    def max_hold_ns(self):
        if isinstance(self.max_hold, str):
            return pd.Timedelta(self.max_hold).value
        return int(self.max_hold) * 10**9 + int(self.max_hold % 1 * 1e9)


@dataclass
class Result:
    """Trade ledger from one replay. ``arrays`` holds the raw numpy columns."""
    arrays: dict = field(repr=False)
    config: Config = field(repr=False)

    def __len__(self):
        return len(self.arrays["dir"])

    @property
    def trades(self):
        a = self.arrays
        df = pd.DataFrame({k: v for k, v in a.items() if k not in
                           ("entry_time_ns", "exit_time_ns", "exit_reason")})
        df["entry_time"] = pd.to_datetime(a["entry_time_ns"], unit="ns")
        df["exit_time"] = pd.to_datetime(a["exit_time_ns"], unit="ns")
        df["exit_reason"] = pd.Series(a["exit_reason"]).map(EXIT_REASONS)
        return df

    def summary(self):
        pnl = self.arrays["pnl_net"]
        if len(pnl) == 0:
            return {"n_trades": 0, "total_net": 0.0, "mean_net": 0.0, "hit_rate": 0.0,
                    "max_drawdown": 0.0}
        cum = np.cumsum(pnl)
        dd = np.maximum.accumulate(cum) - cum
        return {
            "n_trades": len(pnl),
            "total_net": float(cum[-1]),
            "mean_net": float(pnl.mean()),
            "hit_rate": float((pnl > 0).mean()),
            "max_drawdown": float(dd.max()),
        }


@njit(cache=True, nogil=True)
def _replay(time_ns, bid, ask, bar_tick_last, bar_signal, sl_dist, tp_dist,
            allow_buy, allow_sell, max_hold_ns, exit_on_opposite,
            o_dir, o_entry_idx, o_exit_idx, o_entry_price, o_exit_price,
            o_pnl_gross, o_mfe, o_mae, o_reason):
    n_ticks = time_ns.shape[0]
    n_bars = bar_signal.shape[0]
    t = 0
    min_trigger = 0  # post-exit: exit tick + 1 (signals at the exit tick are ignored)
    for b in range(n_bars):
        s = bar_signal[b]
        if s == 0 or (s == 1 and not allow_buy) or (s == -1 and not allow_sell):
            continue
        e = bar_tick_last[b] + 1
        if e < min_trigger:
            continue
        if e >= n_ticks - 1:  # no tick after entry to evaluate -> no trade
            continue
        entry_price = ask[e] if s == 1 else bid[e]
        deadline = time_ns[e] + max_hold_ns
        sl = sl_dist[b]
        tp = tp_dist[b]
        opp_trigger = n_ticks  # sentinel: never reached
        if exit_on_opposite:
            for ob in range(b + 1, n_bars):
                if bar_signal[ob] == -s:
                    opp_trigger = bar_tick_last[ob] + 1
                    break
        mfe = 0.0
        mae = 0.0
        exit_idx = n_ticks - 1
        reason = 5  # eod
        for i in range(e + 1, n_ticks):
            pnl_raw = bid[i] - entry_price if s == 1 else entry_price - ask[i]
            if pnl_raw > mfe:
                mfe = pnl_raw
            if -pnl_raw > mae:
                mae = -pnl_raw
            if sl > 0 and pnl_raw <= -sl:
                exit_idx = i; reason = 1; break
            if tp > 0 and pnl_raw >= tp:
                exit_idx = i; reason = 2; break
            if max_hold_ns > 0 and time_ns[i] >= deadline:
                exit_idx = i; reason = 3; break
            if i == opp_trigger:
                exit_idx = i; reason = 4; break
        exit_price = bid[exit_idx] if s == 1 else ask[exit_idx]
        o_dir[t] = s
        o_entry_idx[t] = e
        o_exit_idx[t] = exit_idx
        o_entry_price[t] = entry_price
        o_exit_price[t] = exit_price
        o_pnl_gross[t] = exit_price - entry_price if s == 1 else entry_price - exit_price
        o_mfe[t] = mfe
        o_mae[t] = mae
        o_reason[t] = reason
        t += 1
        min_trigger = exit_idx + 1
    return t


def replay(store, bars, signal, config=None):
    """Replay ``signal`` (+1 buy / -1 sell / 0 flat, one value per bar) through the
    tick stream. Returns a :class:`Result`."""
    config = config or Config()
    signal = np.ascontiguousarray(signal, dtype=np.int8)
    if len(signal) != len(bars):
        raise ValueError(f"signal has {len(signal)} values for {len(bars)} bars")
    tick_last = np.ascontiguousarray(bars["tick_last"].to_numpy(), np.int64)
    if len(tick_last) > 1 and (np.diff(tick_last) <= 0).any():
        raise ValueError("bars must have strictly increasing tick_last")
    n_bars = len(bars)
    sl = np.ascontiguousarray(np.broadcast_to(np.nan_to_num(config.sl), n_bars),
                              dtype=np.float64)
    tp = np.ascontiguousarray(np.broadcast_to(np.nan_to_num(config.tp), n_bars),
                              dtype=np.float64)
    cap = int((signal != 0).sum()) + 1
    o = {
        "dir": np.zeros(cap, np.int8),
        "entry_idx": np.zeros(cap, np.int64), "exit_idx": np.zeros(cap, np.int64),
        "entry_price": np.zeros(cap, np.float64), "exit_price": np.zeros(cap, np.float64),
        "pnl_gross": np.zeros(cap, np.float64),
        "mfe": np.zeros(cap, np.float64), "mae": np.zeros(cap, np.float64),
        "exit_reason": np.zeros(cap, np.int8),
    }
    time_ns = np.ascontiguousarray(store.time_ns, np.int64)
    n = _replay(time_ns,
                np.ascontiguousarray(store.bid, np.float64),
                np.ascontiguousarray(store.ask, np.float64),
                tick_last,
                signal, sl, tp, config.allow_buy, config.allow_sell,
                np.int64(config.max_hold_ns()), config.exit_on_opposite,
                o["dir"], o["entry_idx"], o["exit_idx"], o["entry_price"],
                o["exit_price"], o["pnl_gross"], o["mfe"], o["mae"], o["exit_reason"])
    arrays = {k: v[:n].copy() for k, v in o.items()}
    arrays["entry_time_ns"] = time_ns[arrays["entry_idx"]]
    arrays["exit_time_ns"] = time_ns[arrays["exit_idx"]]
    arrays["pnl_net"] = arrays["pnl_gross"] - config.cost
    return Result(arrays, config)
