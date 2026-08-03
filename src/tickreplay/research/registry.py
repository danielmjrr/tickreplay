"""Append-only evaluation registry: every evaluation, no exceptions.

The point is multiple-testing honesty. If you run a thousand backtests and report
the best one, its p-value is meaningless; the registry records every attempt, and
q-values are recomputed over the FULL history on read — never stored — so each new
evaluation makes every past q slightly stricter. Keep the CSV in version control.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .stats import bh_qvalues

REQUIRED = ["campaign", "symbol", "config", "n_trades", "mean_net", "t_stat", "p_value"]


def append(path, rows):
    for r in rows:
        missing = [c for c in REQUIRED if c not in r]
        if missing:
            raise ValueError(f"registry row missing {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df = pd.DataFrame([{**r, "ts": ts} for r in rows])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load(path):
    """Full registry with q-values computed over every row ever appended."""
    df = pd.read_csv(path)
    df["q_value"] = bh_qvalues(df["p_value"].to_numpy())
    return df
