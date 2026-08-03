"""Sweeps must fan out without scrambling which result belongs to which config."""
import time

import numpy as np
import pytest

import tickreplay as tr


def crossover_case():
    time_ns, bid, ask = tr.synth.generate_ticks(days=3, ticks_per_day=8000, seed=5)
    store = tr.TickStore(time_ns, bid, ask)
    bars = store.bars("M5")
    trend = np.sign(bars["close"].rolling(5).mean() - bars["close"].rolling(20).mean())
    signal = trend.fillna(0).to_numpy().astype("int8")
    signal[signal == np.roll(signal, 1)] = 0  # act on crosses only
    return store, bars, signal


def test_sweep_exits_returns_one_row_per_config_matching_a_direct_replay():
    store, bars, signal = crossover_case()
    base = {"max_hold": "1h", "cost": 0.09}
    grid = [{"sl": sl, "tp": tp} for sl in (1.0, 3.0) for tp in (2.0, 6.0)]

    df = tr.sweep_exits(store, bars, signal, grid, base=base)

    assert len(df) == len(grid)
    assert df["n_trades"].min() > 0
    assert df["total_net"].nunique() == len(grid), "grid must actually discriminate"
    for row, overrides in zip(df.to_dict("records"), grid):
        assert (row["sl"], row["tp"]) == (overrides["sl"], overrides["tp"])
        expected = tr.replay(store, bars, signal,
                             tr.Config(**base, **overrides)).summary()
        assert set(expected) <= set(row)
        for key, value in expected.items():
            assert row[key] == pytest.approx(value), f"{overrides}: {key}"


def test_sweep_exits_applies_base_under_the_grid():
    store, bars, signal = crossover_case()
    grid = [{"sl": 2.0}]

    with_hold = tr.sweep_exits(store, bars, signal, grid, base={"max_hold": "15m"})
    without_hold = tr.sweep_exits(store, bars, signal, grid)

    assert with_hold.loc[0, "n_trades"] != without_hold.loc[0, "n_trades"]


def test_run_sweep_preserves_config_to_result_pairing():
    # descending delays make workers finish in roughly reverse submission order
    configs = [{"idx": i, "delay": (7 - i) * 0.02} for i in range(8)]

    def evaluate(config):
        time.sleep(config["delay"])
        return {"echo": config["idx"] * 10}

    df = tr.run_sweep(evaluate, configs, workers=4)

    assert list(df["idx"]) == [c["idx"] for c in configs]
    assert list(df["echo"]) == [c["idx"] * 10 for c in configs]


def test_run_sweep_runs_configs_concurrently():
    def evaluate(config):
        time.sleep(0.1)
        return {"idx_seen": config["idx"]}

    configs = [{"idx": i} for i in range(8)]
    started = time.perf_counter()
    df = tr.run_sweep(evaluate, configs, workers=8)
    elapsed = time.perf_counter() - started

    assert list(df["idx_seen"]) == list(range(8))
    assert elapsed < 0.4, f"8 x 100ms across 8 workers took {elapsed:.2f}s -- serialized?"
