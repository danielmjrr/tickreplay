# tickreplay

[![ci](https://github.com/danielmjrr/tickreplay/actions/workflows/ci.yml/badge.svg)](https://github.com/danielmjrr/tickreplay/actions/workflows/ci.yml)

Backtest on every tick instead of on bars — at roughly a billion ticks per second.
tickreplay pairs a memory-mapped tick store and a JIT-compiled replay kernel with the
statistical tooling to tell a real edge from a backtesting artifact. Years of MT5
tick history replay in milliseconds; a hundred-config parameter sweep finishes before
your coffee does.

It is built around two ideas:

1. **Bar backtests lie.** Inside one OHLC bar you cannot know whether the high or the
   low printed first, so any bar-level simulation of a stop loss and take profit is
   guessing about path. tickreplay replays every tick, so barrier ordering, spread
   costs, and fill prices are exact.
2. **Most backtest profits are statistical artifacts.** The library ships the
   machinery to catch that: purged walk-forward splits, overlap-corrected
   significance tests, and an append-only evaluation registry with cumulative
   false-discovery control.

It is fast enough that neither idea costs you iteration speed: the replay kernel
streams a year of ticks in well under a second on a laptop.

## Install

```
pip install git+https://github.com/danielmjrr/tickreplay
```

Python ≥ 3.11. Dependencies: numpy, pandas, numba, pyarrow, scipy.

## Quickstart

```python
import numpy as np
import tickreplay as tr

# synthetic data so this runs without any broker export
time_ns, bid, ask = tr.synth.generate_ticks(days=30, seed=7)
store = tr.TickStore(time_ns, bid, ask)
bars = store.bars("M5")

# a signal is one int per bar: +1 buy, -1 sell, 0 nothing
fast = bars["close"].rolling(10).mean()
slow = bars["close"].rolling(50).mean()
above = (fast > slow).astype("int8")
signal = np.where(above.diff() > 0, 1, np.where(above.diff() < 0, -1, 0))

result = tr.replay(store, bars, signal,
                   tr.Config(sl=3.0, tp=6.0, max_hold="4h", cost=0.05))
print(result.summary())
print(result.trades.head())
```

Entries happen at the first tick after the signal bar closes (buys at ask, sells at
bid); exits are checked tick-by-tick in the order stop → target → time → opposite
signal. One position at a time — the result is the *tradable* expectancy of the
strategy, not an overlapped ideal. Full semantics are documented in
`tickreplay/engine.py`.

## Using your own data

Export ticks from MT5 (View → Symbols → select symbol → Ticks tab → request a range →
Export). That produces a tab-separated CSV. Then:

```python
import tickreplay as tr

tr.ingest.ticks_to_monthly_parquet("XAUUSD_ticks.csv", "data/xauusd/parquet")
store = tr.TickStore.build("data/xauusd/parquet", "data/xauusd/store")
# later sessions:
store = tr.TickStore.open("data/xauusd/store")
```

The store is a set of memory-mapped flat arrays: opening it is instant, the OS page
cache shares it across processes, and a years-long tick history costs no RAM until
touched. `tr.ingest.validate_against_broker_bars` cross-checks the tick export
against the broker's own M1 bars — recommended before trusting any new export
(mismatches usually mean the tick and bar files came from different servers).

## Why it's fast

- **Storage**: ticks live in memory-mapped `int64/float64` arrays; bars carry their
  tick index spans, so mapping a bar to its ticks is O(1) and derived data never
  duplicates the tick stream.
- **Kernel**: the replay loop is a single numba-JIT function, `nogil` and
  branch-light, that streams the arrays once. It is memory-bandwidth-bound — on an
  Apple M4 it processes roughly a billion ticks per second.
- **Sweeps**: because the kernel releases the GIL and the data is shared, parameter
  grids fan out over threads with zero copies (`tr.sweep_exits`). A 100-config grid
  over years of ticks takes seconds.

Run `python examples/benchmark.py` for numbers on your machine.

## Why you can trust it

- **Two independent engines.** The numba kernel is tested trade-for-trade,
  byte-for-byte against a deliberately naive pure-Python reference implementation
  (`tests/reference.py`) across randomized fixtures and config combinations. Any
  semantic drift between the two fails CI.
- **Look-ahead is tested, not assumed.** Truncation tests verify that removing
  future data leaves all earlier trades byte-identical; bar spans make the
  no-look-ahead boundary explicit (a bar exists only after its last tick).
- **The research module assumes you will fool yourself** and gets in the way:
  - `research.validation.purged_expanding_folds` — walk-forward splits that purge
    training samples whose label window leaks into the test period (plus embargo);
  - `research.stats.thin_by_gap` — overlapping trade windows inflate t-statistics;
    test a non-overlapping subset instead;
  - `research.registry` — an append-only log of *every* evaluation you ever ran,
    with Benjamini-Hochberg q-values recomputed over the full history on every
    read. Run a thousand configs and the bar for "significant" rises accordingly;
  - `research.labels` — tick-exact excursion and first-passage labels for
    measurement (where bar data cannot even order target-vs-stop correctly);
  - `research.bocpd` — online Bayesian changepoint detection (Adams & MacKay),
    strictly filtered, with causality enforced by test.

`examples/sweep_and_validate.py` shows the intended workflow end to end — including
what an honest negative result looks like.

## Examples

All examples run on synthetic data in about a second, no broker export needed:

- [`examples/ma_cross.py`](examples/ma_cross.py) — moving-average cross, fixed stops
- [`examples/breakout.py`](examples/breakout.py) — N-bar breakout with
  opposite-signal exits and an exit-reason breakdown
- [`examples/sweep_and_validate.py`](examples/sweep_and_validate.py) — the intended
  research loop: sweep a grid, then test the winner with thinning and a
  selection-adjusted threshold
- [`examples/benchmark.py`](examples/benchmark.py) — throughput measurement on your
  hardware (on an M4 MacBook Air: ~1,100M ticks/s scanned inside positions, 20M-tick
  store replayed in under 4 ms)

## Scope and limitations

- Single position, netting-style: no pyramiding, no hedged legs, no partial fills.
- Quote data only (bid/ask). Spot FX/metals CFDs have no real volume feed; if your
  strategy needs volume, you need futures data instead.
- Costs = spread (paid via bid/ask fills) + a fixed round-trip add. No market-impact
  model — realistic for retail sizes, not for size.
- Timestamps are broker time; the library never converts time zones.

## License

MIT
