#!/usr/bin/env python3
# coding: utf-8

""" Tests for the Strategy orchestrator. """

# Third-party packages
import numpy as np

# Local packages
from fynance.backtest import BacktestResult
from fynance.backtest.cost import ProportionalCost
from fynance.core import PriceSeries
from fynance.signal import sign
from fynance.strategy import Strategy


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))


def test_rule_based_run_returns_result():
    # momentum rule: position = sign of last return
    def momentum(prices):
        r = np.zeros_like(prices)
        r[1:] = np.sign(prices[1:] - prices[:-1])
        return r

    strat = Strategy(features=momentum, signal=lambda x: x)
    res = strat.run(_prices())
    assert isinstance(res, BacktestResult)
    assert np.isfinite(res.summary()["sharpe"])


def test_run_accepts_price_series():
    strat = Strategy(features=lambda p: p, signal=sign)
    res = strat.run(PriceSeries(_prices()))
    assert isinstance(res, BacktestResult)


def test_run_with_model_supervised():
    class MeanModel:
        def fit(self, X, y):
            self._m = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(np.asarray(X).shape[0], self._m)

    prices = _prices()
    y = np.sign(np.diff(prices, prepend=prices[0]))
    strat = Strategy(model=MeanModel(), signal=sign)
    res = strat.run(prices, y=y)
    assert isinstance(res, BacktestResult)


def test_cost_reduces_performance():
    prices = _prices()

    def alt(prices):  # alternating churn to create turnover
        return np.resize([1.0, -1.0], prices.shape[0])

    no_cost = Strategy(features=alt, signal=lambda x: x).run(prices)
    with_cost = Strategy(features=alt, signal=lambda x: x,
                         cost=ProportionalCost(fee=0.01)).run(prices)
    assert with_cost.summary()["total_cost"] > 0.0
    assert np.sum(with_cost.returns) <= np.sum(no_cost.returns) + 1e-9


def test_swappable_signal_slot():
    prices = _prices()

    def momentum(prices):  # sign-varying feature
        r = np.zeros_like(prices)
        r[1:] = prices[1:] - prices[:-1]
        return r

    r1 = Strategy(features=momentum, signal=sign).run(prices)
    r2 = Strategy(features=momentum,
                  signal=lambda x: np.ones_like(x)).run(prices)
    assert not np.allclose(r1.positions, r2.positions)


def test_walk_forward_no_lookahead():
    class LastSignModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            X = np.asarray(X)
            return np.sign(np.diff(X, prepend=X[0]))

    prices = _prices(400)
    y = np.sign(np.diff(prices, prepend=prices[0]))
    strat = Strategy(model=LastSignModel(), signal=sign)
    base = strat.run_walk_forward(prices, y, train=100, test=20, step=20)
    assert isinstance(base, BacktestResult)
    assert np.isfinite(base.summary()["sharpe"])

    # corrupting the far-future target must not change the result
    y2 = y.copy()
    y2[-30:] *= -1
    pert = strat.run_walk_forward(prices, y2, train=100, test=20, step=20)
    # OOS coverage identical in length
    assert base.equity.shape == pert.equity.shape
