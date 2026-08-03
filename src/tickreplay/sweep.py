"""Parallel parameter sweeps.

Threads, not processes: the replay kernel releases the GIL and the tickstore is
memory-mapped, so workers share one copy of the data and scale near-linearly with
zero IPC. A full parameter grid over years of ticks runs in seconds.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .engine import Config, replay


def run_sweep(evaluate, configs, workers=None):
    """Map ``evaluate`` over configs concurrently; returns a DataFrame with one row
    per config (config keys + whatever dict ``evaluate`` returns)."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(evaluate, configs))
    return pd.DataFrame([{**c, **r} for c, r in zip(configs, results)])


def sweep_exits(store, bars, signal, grid, base=None, workers=None):
    """Sweep exit-rule variants for one signal: ``grid`` is a list of dicts of
    :class:`Config` field overrides applied on top of ``base``."""
    base = base or {}

    def evaluate(overrides):
        return replay(store, bars, signal, Config(**{**base, **overrides})).summary()

    return run_sweep(evaluate, grid, workers=workers)
