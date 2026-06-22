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


def test_run_with_precomputed_features_X():
    # X replaces features(prices): the model never sees the price-only
    # featurizer, so a constant exogenous feature must drive the position.
    prices = _prices(120)

    class PassThrough:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.asarray(X, dtype=np.float64).reshape(-1)

    # Exogenous feature: +1 everywhere -> sign() -> long everywhere.
    X = np.ones(prices.shape[0])
    strat = Strategy(model=PassThrough(), signal=sign)
    res = strat.run(prices, X=X)
    assert isinstance(res, BacktestResult)
    # All positions long (1.0) since X is constant +1.
    assert np.allclose(res.positions, 1.0)

    # And X actually overrides the featurizer: flipping X flips the book.
    res_neg = strat.run(prices, X=-np.ones(prices.shape[0]))
    assert np.allclose(res_neg.positions, -1.0)


def test_run_walk_forward_with_precomputed_features_X():
    prices = _prices(300)
    y = np.sign(np.diff(prices, prepend=prices[0]))

    class MeanModel:
        def fit(self, X, y):
            self._m = float(np.mean(np.asarray(y)))
            return self

        def predict(self, X):
            return np.full(np.asarray(X).shape[0], self._m)

    # 2-D exogenous feature matrix aligned with the price index.
    X = np.column_stack([y, np.ones_like(y)])
    strat = Strategy(model=MeanModel(), signal=sign)
    res = strat.run_walk_forward(prices, y, train=100, test=20, step=20, X=X)
    assert isinstance(res, BacktestResult)
    assert np.isfinite(res.summary()["sharpe"])

    # No-lookahead with X: perturbing the FUTURE leaves earlier OOS positions
    # untouched (X is sliced per window, never globally).
    cut = len(prices) - 40
    p2 = prices.copy()
    p2[cut:] *= 1.2
    y2 = np.sign(np.diff(p2, prepend=p2[0]))
    X2 = np.column_stack([y2, np.ones_like(y2)])
    pert = strat.run_walk_forward(p2, y2, train=100, test=20, step=20, X=X2)
    prefix = cut - 100 - 40
    assert prefix > 0
    assert np.allclose(res.positions[:prefix], pert.positions[:prefix])


def test_walk_forward_no_lookahead():
    # A model that *genuinely depends on y* (so the probe is not vacuous).
    class MeanSignModel:
        def fit(self, X, y):
            self.bias = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(np.asarray(X).shape[0], self.bias)

    prices = _prices(400)
    y = np.sign(np.diff(prices, prepend=prices[0]))
    strat = Strategy(model=MeanSignModel(), signal=sign)
    base = strat.run_walk_forward(prices, y, train=100, test=20, step=20)
    assert isinstance(base, BacktestResult)
    assert np.isfinite(base.summary()["sharpe"])

    # No-lookahead: perturbing the FUTURE (last 40 obs of prices and targets)
    # must leave every out-of-sample position that predates it unchanged.
    cut = len(prices) - 40
    p2 = prices.copy()
    p2[cut:] *= 1.2
    y2 = np.sign(np.diff(p2, prepend=p2[0]))
    pert = strat.run_walk_forward(p2, y2, train=100, test=20, step=20)

    assert base.positions.shape == pert.positions.shape
    # positions whose window ends before the perturbation are identical
    prefix = cut - 100 - 40  # conservative: before any window touching the tail
    assert prefix > 0
    assert np.allclose(base.positions[:prefix], pert.positions[:prefix])
