#!/usr/bin/env python3
# coding: utf-8

""" Tests for BacktestResult. """

# Third-party packages
import numpy as np

# Local packages
from fynance.backtest import backtest
from fynance.core import PriceSeries


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
