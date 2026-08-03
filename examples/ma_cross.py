"""Moving-average cross on M5 bars, replayed tick by tick.

Run: python examples/ma_cross.py
"""
import numpy as np

import tickreplay as tr


def main():
    time_ns, bid, ask = tr.synth.generate_ticks(days=30, seed=7)
    store = tr.TickStore(time_ns, bid, ask)
    bars = store.bars("M5")

    fast = bars["close"].rolling(10).mean()
    slow = bars["close"].rolling(50).mean()

    # Regime is +1 while fast is above slow. The signal fires only on the bar where
    # the regime flips, so one cross is one entry instead of one entry per bar.
    regime = np.sign(fast - slow).fillna(0.0).to_numpy()
    previous = np.r_[0.0, regime[:-1]]
    signal = np.where(regime != previous, regime, 0).astype("int8")

    # signal[b] is built from closes up to and including bar b, so it is knowable
    # only once bar b has closed. No extra shift is needed: the engine enters at the
    # first tick after the signal bar closes.
    config = tr.Config(sl=3.0, tp=6.0, max_hold="4h", cost=0.05)
    result = tr.replay(store, bars, signal, config)

    print(f"ticks     {len(store):>12,}")
    print(f"M5 bars   {len(bars):>12,}")
    print(f"signals   {int((signal != 0).sum()):>12,}")
    print("config    sl=3.0 tp=6.0 max_hold=4h cost=0.05 (price units)")
    print()
    for key, value in result.summary().items():
        print(f"{key:<14}{value:>12.4f}" if isinstance(value, float)
              else f"{key:<14}{value:>12,}")


if __name__ == "__main__":
    main()
