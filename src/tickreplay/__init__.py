"""tickreplay: tick-accurate backtesting for MT5 data.

Quickstart::

    import numpy as np
    import tickreplay as tr

    store = tr.TickStore.open("data/my_symbol")   # built once from an MT5 export
    bars = store.bars("M5")

    fast = bars.close.rolling(10).mean()
    slow = bars.close.rolling(50).mean()
    signal = np.sign(fast - slow).fillna(0).astype("int8")
    signal[signal == np.roll(signal, 1)] = 0      # act on crosses only

    result = tr.replay(store, bars, signal,
                       tr.Config(sl=5.0, tp=10.0, max_hold="4h", cost=0.09))
    print(result.summary())
"""
from . import ingest, research, synth
from .engine import Config, Result, replay
from .store import TIMEFRAMES, TickStore, derive_bars
from .sweep import run_sweep, sweep_exits

__version__ = "0.1.0"
__all__ = ["Config", "Result", "TickStore", "TIMEFRAMES", "derive_bars", "ingest",
           "replay", "research", "run_sweep", "sweep_exits", "synth"]
