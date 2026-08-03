"""Replay throughput on synthetic ticks.

Run: python examples/benchmark.py [--ticks 20000000]
"""
import argparse
import statistics
import time

import numpy as np

import tickreplay as tr

TICKS_PER_DAY = 200_000
WARMUP_TICKS = 100_000


def build_store(n_ticks, seed):
    # generate_ticks skips weekends, so ask for enough calendar days to cover it.
    days = int(np.ceil(n_ticks / TICKS_PER_DAY * 7 / 5)) + 3
    time_ns, bid, ask = tr.synth.generate_ticks(
        days=days, ticks_per_day=TICKS_PER_DAY, seed=seed)
    return tr.TickStore(time_ns[:n_ticks], bid[:n_ticks], ask[:n_ticks])


def ma_cross_signal(bars):
    fast = bars["close"].rolling(10).mean()
    slow = bars["close"].rolling(50).mean()
    regime = np.sign(fast - slow).fillna(0.0).to_numpy()
    return np.where(regime != np.r_[0.0, regime[:-1]], regime, 0).astype("int8")


def main():
    parser = argparse.ArgumentParser(description="Replay throughput on synthetic ticks.")
    parser.add_argument("--ticks", type=int, default=20_000_000,
                        help="store size; 20M ticks needs about 1 GB of RAM")
    parser.add_argument("--timeframe", default="M5", choices=sorted(tr.TIMEFRAMES),
                        help="bar timeframe the signal is built on")
    parser.add_argument("--seed", type=int, default=1, help="synthetic data seed")
    args = parser.parse_args()

    t0 = time.perf_counter()
    store = build_store(args.ticks, args.seed)
    gen_ms = (time.perf_counter() - t0) * 1e3

    config = tr.Config(sl=3.0, tp=6.0, max_hold="4h", cost=0.05)

    # Compile the kernel on a small slice so JIT time stays out of the measurement.
    warm = tr.TickStore(store.time_ns[:WARMUP_TICKS], store.bid[:WARMUP_TICKS],
                        store.ask[:WARMUP_TICKS])
    warm_bars = warm.bars(args.timeframe)
    tr.replay(warm, warm_bars, ma_cross_signal(warm_bars), config)

    t0 = time.perf_counter()
    bars = store.bars(args.timeframe)
    bars_ms = (time.perf_counter() - t0) * 1e3
    signal = ma_cross_signal(bars)

    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = tr.replay(store, bars, signal, config)
        timings.append((time.perf_counter() - t0) * 1e3)
    median_ms = statistics.median(timings)

    # The kernel only walks ticks while a position is open, so quote the covered
    # fraction alongside the headline rate.
    visited = int((result.arrays["exit_idx"] - result.arrays["entry_idx"]).sum())

    print(f"ticks            {len(store):>14,}")
    print(f"{args.timeframe} bars{'':<10}{len(bars):>14,}")
    print(f"trades           {len(result):>14,}")
    print(f"ticks in position{visited:>14,}  ({visited / max(len(store), 1):.1%})")
    print(f"generate         {gen_ms:>14,.1f} ms")
    print(f"derive bars      {bars_ms:>14,.1f} ms")
    print(f"replay (median)  {median_ms:>14,.1f} ms  of {[round(x, 1) for x in timings]}")
    print(f"kernel rate      {visited / median_ms / 1e3:>14,.1f} M ticks/s scanned")
    print(f"end-to-end rate  {len(store) / median_ms / 1e3:>14,.1f} M store ticks/s")


if __name__ == "__main__":
    main()
