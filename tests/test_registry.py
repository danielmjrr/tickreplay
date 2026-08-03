import numpy as np
import pytest

from tickreplay.research.registry import REQUIRED, append, load
from tickreplay.research.stats import bh_qvalues


def row(config, p_value, **over):
    base = {"campaign": "c1", "symbol": "SYNTH", "config": config, "n_trades": 120,
            "mean_net": 0.02, "t_stat": 2.0, "p_value": p_value}
    base.update(over)
    return base


def test_append_rejects_rows_missing_required_columns(tmp_path):
    path = tmp_path / "registry.csv"
    incomplete = row("a", 0.01)
    del incomplete["p_value"]

    with pytest.raises(ValueError, match="p_value"):
        append(path, [incomplete])

    assert not path.exists()


def test_append_validates_every_row_before_writing_any(tmp_path):
    path = tmp_path / "registry.csv"
    bad = row("b", 0.01)
    del bad["n_trades"]

    with pytest.raises(ValueError):
        append(path, [row("a", 0.01), bad])

    assert not path.exists()


@pytest.mark.parametrize("column", REQUIRED)
def test_every_required_column_is_enforced(tmp_path, column):
    path = tmp_path / f"registry_{column}.csv"
    incomplete = row("a", 0.01)
    del incomplete[column]

    with pytest.raises(ValueError, match=column):
        append(path, [incomplete])


def test_append_twice_then_load_returns_all_rows_with_qvalues(tmp_path):
    path = tmp_path / "nested" / "registry.csv"

    append(path, [row("a", 0.01), row("b", 0.50)])
    append(path, [row("c", 0.60), row("d", 0.70)])

    df = load(path)

    assert len(df) == 4
    assert list(df["config"]) == ["a", "b", "c", "d"]
    assert "q_value" in df.columns
    assert "ts" in df.columns
    assert set(REQUIRED) <= set(df.columns)
    np.testing.assert_allclose(df["q_value"].to_numpy(),
                               bh_qvalues(df["p_value"].to_numpy()))


def test_qvalues_are_recomputed_cumulatively_not_stored(tmp_path):
    path = tmp_path / "registry.csv"

    append(path, [row("a", 0.01), row("b", 0.50)])
    first = load(path)
    assert first.loc[0, "q_value"] == pytest.approx(0.02)
    assert first.loc[1, "q_value"] == pytest.approx(0.50)

    append(path, [row("c", 0.60), row("d", 0.70)])
    second = load(path)

    assert second.loc[0, "q_value"] == pytest.approx(0.04)
    assert second.loc[1, "q_value"] == pytest.approx(0.70)
    assert second.loc[0, "q_value"] > first.loc[0, "q_value"]
    assert "q_value" not in path.read_text().splitlines()[0]


def test_registry_is_append_only(tmp_path):
    path = tmp_path / "registry.csv"

    append(path, [row("a", 0.01)])
    append(path, [row("a", 0.01)])

    df = load(path)

    assert len(df) == 2
    assert path.read_text().count("campaign,symbol") == 1
