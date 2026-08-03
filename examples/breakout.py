"""N-bar breakout on M5 bars, exiting on the opposite breakout.

Run: python examples/breakout.py
"""
import numpy as np

import tickreplay as tr

LOOKBACK = 20


def main():
    time_ns, bid, ask = tr.synth.generate_ticks(days=30, seed=11)
    store = tr.TickStore(time_ns, bid, ask)
    bars = store.bars("M5")

    # shift(1) keeps the current bar out of its own reference range.
    prior_high = bars["high"].shift(1).rolling(LOOKBACK).max()
    prior_low = bars["low"].shift(1).rolling(LOOKBACK).min()
    state = np.where(bars["close"] > prior_high, 1,
                     np.where(bars["close"] < prior_low, -1, 0)).astype("int8")

    # Fire on the bar that breaks out, not on every bar that stays beyond the range.
    signal = np.where(state != np.r_[np.int8(0), state[:-1]], state, 0).astype("int8")

    config = tr.Config(sl=4.0, tp=8.0, max_hold="8h", exit_on_opposite=True, cost=0.05)
    result = tr.replay(store, bars, signal, config)

    print(f"ticks     {len(store):>12,}")
    print(f"M5 bars   {len(bars):>12,}")
    print(f"signals   {int((signal != 0).sum()):>12,}  ({LOOKBACK}-bar breakout)")
    print("config    sl=4.0 tp=8.0 max_hold=8h exit_on_opposite=True cost=0.05")
    print()
    for key, value in result.summary().items():
        print(f"{key:<14}{value:>12.4f}" if isinstance(value, float)
              else f"{key:<14}{value:>12,}")

    print()
    print("exit reasons")
    counts = result.trades["exit_reason"].value_counts()
    for reason, count in counts.items():
        print(f"  {reason:<10}{count:>6,}{count / counts.sum():>8.1%}")


if __name__ == "__main__":
    main()
