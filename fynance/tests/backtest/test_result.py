#!/usr/bin/env python3
# coding: utf-8

""" Tests for BacktestResult. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.backtest import backtest
from fynance.core import PriceSeries
from fynance.metrics.trades import TRADE_DTYPE, extract_trades, trade_summary


def test_summary_keys_and_finiteness():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0005, 0.01, 300)
    positions = np.sign(rng.normal(size=300))
    res = backtest(returns, positions, shift=True)
    s = res.summary()
    for key in ("annual_return", "annual_volatility", "sharpe", "sortino",
                "max_drawdown", "calmar", "hit_rate", "total_cost",
                "n_sign_changes", "trades_per_year"):
        assert key in s
        assert np.isfinite(s[key])


def test_summary_trade_profile_counts_sign_changes():
    # Positions flip direction every bar -> a sign change at each step.
    positions = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    res = backtest(np.zeros(6), positions, shift=False)
    s = res.summary(period=252)
    assert s["n_sign_changes"] == 5.0
    assert np.isclose(s["trades_per_year"], 5.0 / 6.0 * 252.0)


def test_to_price_series():
    res = backtest(np.array([0.01, 0.02, -0.01]), np.ones(3))
    eq = res.to_price_series()
    assert isinstance(eq, PriceSeries)
    assert np.allclose(eq.values, res.equity)


def test_max_drawdown_nonnegative():
    res = backtest(np.array([0.1, -0.5, 0.2]), np.ones(3), shift=False)
    assert res.summary()["max_drawdown"] >= 0.0


def test_trades_matches_standalone_extract_trades():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, 200)
    positions = rng.choice([-1.0, 0.0, 1.0], size=200, p=[0.3, 0.2, 0.5])
    res = backtest(returns, positions, shift=False)

    out = res.trades()
    ref = extract_trades(res.positions, res.returns)

    assert out.dtype == TRADE_DTYPE
    assert np.array_equal(out, ref)


def test_trade_summary_matches_standalone_trade_summary():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.01, 200)
    positions = rng.choice([-1.0, 0.0, 1.0], size=200, p=[0.3, 0.2, 0.5])
    res = backtest(returns, positions, shift=False)

    out = res.trade_summary()
    ref = trade_summary(res.trades())

    assert out.keys() == ref.keys()
    for key in out:
        if np.isnan(ref[key]):
            assert np.isnan(out[key])
        else:
            assert out[key] == pytest.approx(ref[key])


def test_trades_empty_positions_edge():
    res = backtest(np.zeros(10), np.zeros(10), shift=False)
    out = res.trades()
    assert out.shape[0] == 0
    assert res.trade_summary()["n_trades"] == 0.0


# -- to_pandas ---------------------------------------------------------------

def test_to_pandas_single_asset_columns_shape_dtypes():
    pd = pytest.importorskip("pandas")
    res = backtest(np.array([0.01, 0.02, -0.01]), np.ones(3), shift=False)
    df = res.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "equity", "returns", "gross_returns", "costs", "positions",
    ]
    assert df.shape == (3, 5)
    for col in df.columns:
        assert df[col].dtype == np.float64
    assert np.array_equal(df["equity"].to_numpy(), res.equity)
    assert np.array_equal(df["positions"].to_numpy(), res.positions)


def test_to_pandas_uses_index_when_set():
    pytest.importorskip("pandas")
    res = backtest(np.array([0.01, 0.02, -0.01]), np.ones(3), shift=False)
    res.index = np.array([10, 11, 12])
    df = res.to_pandas()
    assert np.array_equal(df.index.to_numpy(), [10, 11, 12])


def test_to_pandas_multi_asset_positions_become_pos_columns():
    pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 0.01, (20, 3))
    positions = rng.choice([-1.0, 0.0, 1.0], size=(20, 3))
    res = backtest(data, positions, shift=True)
    df = res.to_pandas()

    assert "pos_0" in df.columns and "pos_1" in df.columns and "pos_2" in df.columns
    assert "positions" not in df.columns
    assert np.array_equal(df["pos_0"].to_numpy(), res.positions[:, 0])
    assert np.array_equal(df["pos_1"].to_numpy(), res.positions[:, 1])
    assert np.array_equal(df["pos_2"].to_numpy(), res.positions[:, 2])

    # asset_gross_returns is populated for a multi-asset book
    assert res.asset_gross_returns is not None
    assert "asset_gross_return_0" in df.columns
    assert np.array_equal(
        df["asset_gross_return_0"].to_numpy(), res.asset_gross_returns[:, 0]
    )


def test_to_pandas_cost_components_become_cost_columns():
    pytest.importorskip("pandas")

    class _CostWithComponents:
        def __call__(self, positions):
            return np.abs(positions) * 0.01

        def components(self, positions):
            return {"transaction": np.abs(positions) * 0.01}

    res = backtest(
        np.array([0.01, 0.02, -0.01]), np.ones(3), cost=_CostWithComponents(),
        shift=False,
    )
    df = res.to_pandas()
    assert "cost_transaction" in df.columns
    assert np.array_equal(
        df["cost_transaction"].to_numpy(), res.cost_components["transaction"]
    )


def test_to_pandas_lazy_import_error_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "pandas":

            raise ImportError("No module named 'pandas'")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    res = backtest(np.array([0.01, 0.02, -0.01]), np.ones(3))
    with pytest.raises(ImportError, match="install pandas to use to_pandas"):
        res.to_pandas()
