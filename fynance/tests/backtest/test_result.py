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
                "max_drawdown", "calmar", "hit_rate", "total_cost"):
        assert key in s
        assert np.isfinite(s[key])


def test_to_price_series():
    res = backtest(np.array([0.01, 0.02, -0.01]), np.ones(3))
    eq = res.to_price_series()
    assert isinstance(eq, PriceSeries)
    assert np.allclose(eq.values, res.equity)


def test_max_drawdown_nonnegative():
    res = backtest(np.array([0.1, -0.5, 0.2]), np.ones(3), shift=False)
    assert res.summary()["max_drawdown"] >= 0.0
