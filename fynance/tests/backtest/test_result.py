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
