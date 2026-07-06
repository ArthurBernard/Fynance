#!/usr/bin/env python3
# coding: utf-8

""" Tests for the capacity analysis (net Sharpe vs AUM, breakeven fee). """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.backtest.capacity import breakeven_fee, capacity_curve
from fynance.backtest.cost import MarketImpactCost, ProportionalCost
from fynance.backtest.engine import backtest


def _planted_strategy(seed: int = 0, T: int = 500):
    """ Single-asset weights/prices with a real, causal, tradeable edge.

    ``weights`` is a bounded noisy signal (turnover ~5% per bar from its
    increments); ``returns`` bakes a small edge correlated with the *lagged*
    weight (so ``backtest``'s own one-step shift realizes it), plus noise.
    """
    rng = np.random.default_rng(seed)
    steps = np.cumsum(rng.normal(0.0, 0.07, size=T))
    weights = np.clip(steps, -1.0, 1.0)
    noise = rng.normal(0.0, 0.01, size=T)
    returns = np.empty(T)
    returns[0] = noise[0]
    returns[1:] = 0.0006 * weights[:-1] + noise[1:]
    prices = 100.0 * np.cumprod(1.0 + returns)

    return weights, prices


# -- capacity_curve ---------------------------------------------------------


def test_zero_cost_factory_net_sharpe_equals_gross():
    weights, prices = _planted_strategy(seed=1)
    aums = np.array([1e5, 1e7, 1e9])
    gross_sharpe = backtest(
        prices, weights, returns_input=False
    ).summary()["sharpe"]

    def zero_cost_factory(aum):
        return ProportionalCost(fee=0.0)

    out = capacity_curve(weights, prices, aums, zero_cost_factory)

    assert np.allclose(out["net_sharpe"], gross_sharpe)


def test_convex_impact_net_sharpe_non_increasing_in_aum():
    weights, prices = _planted_strategy(seed=7, T=2000)
    aums = np.logspace(5, 9, 9)
    def impact_factory(aum):
        return MarketImpactCost(impact=0.0017 * np.sqrt(aum / 1e6), exponent=1.5)

    out = capacity_curve(weights, prices, aums, impact_factory)

    # non-increasing (allow tiny numeric slack)
    assert np.all(np.diff(out["net_sharpe"]) <= 1e-9)
    # cost itself must strictly grow with AUM (the driver of the decay)
    assert np.all(np.diff(out["total_cost"]) > 0.0)


def test_output_keys_and_lengths():
    weights, prices = _planted_strategy(seed=2)
    aums = np.array([1e6, 1e7, 1e8, 1e9])

    def factory(aum):
        return ProportionalCost(fee=1e-4)

    out = capacity_curve(weights, prices, aums, factory)

    assert set(out) == {"aum", "net_sharpe", "net_annual_return", "total_cost"}
    for key in out:
        assert out[key].shape == aums.shape


def test_aums_not_1d_raises():
    weights, prices = _planted_strategy(seed=3)
    bad_aums = np.array([[1e5, 1e6]])

    def factory(aum):
        return ProportionalCost(fee=0.0)

    with pytest.raises(ValueError):
        capacity_curve(weights, prices, bad_aums, factory)


def test_shape_mismatch_raises():
    weights = np.ones(10)
    prices = np.linspace(100.0, 110.0, 12)

    def factory(aum):
        return ProportionalCost(fee=0.0)

    with pytest.raises(ValueError):
        capacity_curve(weights, prices, np.array([1e6]), factory)


# -- breakeven_fee ------------------------------------------------------


def test_breakeven_fee_rerun_gives_near_zero_sharpe():
    weights, prices = _planted_strategy(seed=4, T=1000)
    fee = breakeven_fee(weights, prices)
    res = backtest(
        prices, weights, cost=ProportionalCost(fee=fee), returns_input=False
    )
    net_sharpe = res.summary()["sharpe"]
    assert abs(net_sharpe) < 1e-2


def test_breakeven_fee_unprofitable_gross_raises():
    weights, prices = _planted_strategy(seed=5, T=1000)
    # Trading against the planted edge is unprofitable gross by construction.
    with pytest.raises(ValueError, match="unprofitable gross"):
        breakeven_fee(-weights, prices)
