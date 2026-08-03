"""Sweep exit rules, then test the winner honestly.

Picking the best cell of a grid and reporting its p-value as-is is the standard way
to manufacture an edge. This example sweeps, then thins overlapping trades before
testing, and reads the result against a selection-adjusted threshold.

Run: python examples/sweep_and_validate.py
"""
import numpy as np

import tickreplay as tr

HOUR_NS = 3_600 * 10**9


def main():
    time_ns, bid, ask = tr.synth.generate_ticks(days=60, seed=3)
    store = tr.TickStore(time_ns, bid, ask)
    bars = store.bars("M5")

    fast = bars["close"].rolling(10).mean()
    slow = bars["close"].rolling(50).mean()
    regime = np.sign(fast - slow).fillna(0.0).to_numpy()
    signal = np.where(regime != np.r_[0.0, regime[:-1]], regime, 0).astype("int8")

    base = {"max_hold": "4h", "cost": 0.05}
    grid = [{"sl": sl, "tp": tp} for sl in (2.0, 3.0, 4.0) for tp in (4.0, 6.0, 8.0)]
    table = tr.sweep_exits(store, bars, signal, grid, base=base)
    table = table.sort_values("total_net", ascending=False).reset_index(drop=True)

    print(f"{len(grid)} exit configs over {len(store):,} ticks\n")
    print(table[["sl", "tp", "n_trades", "total_net", "hit_rate",
                 "max_drawdown"]].to_string(index=False, float_format="%.3f"))

    best = table.iloc[0]
    result = tr.replay(store, bars, signal,
                       tr.Config(sl=best["sl"], tp=best["tp"], **base))
    pnl = result.arrays["pnl_net"]

    # Trades whose holding windows overlap are correlated; a t-stat over all of them
    # is inflated. Test a non-overlapping subset spaced by the max hold time.
    keep = tr.research.stats.thin_by_gap(result.arrays["entry_time_ns"], 4 * HOUR_NS)
    t_stat, p_value = tr.research.stats.one_sided_positive(pnl[keep])
    threshold = 0.05 / len(grid)

    print(f"\nbest by total_net: sl={best['sl']} tp={best['tp']}")
    print(f"  trades              {len(pnl):>8,} -> {int(keep.sum()):,} after thinning")
    print(f"  one-sided t         {t_stat:>8.3f}")
    print(f"  p (mean > 0)        {p_value:>8.4f}")
    print(f"  threshold (0.05/{len(grid)}) {threshold:>8.4f}")
    print(f"\n  verdict: {'NOT significant' if p_value > threshold else 'passes'}"
          " once selection over the grid is accounted for.")
    print("  Expected: this is a synthetic random walk with costs, so there is no edge"
          "\n  to find. A pass here would mean the procedure is broken, not that the"
          "\n  strategy works.")


if __name__ == "__main__":
    main()
