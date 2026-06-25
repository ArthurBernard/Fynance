#!/usr/bin/env python3
# coding: utf-8

""" Tests for transaction cost models. """

# Third-party packages
import numpy as np
import pytest

from fynance.backtest.cost import MarketImpactCost, ProportionalCost

# Local packages
from fynance.backtest.engine import backtest
from fynance.core import CostModel
from fynance.portfolio.sizing import transaction_cost


def test_zero_fee_zero_cost():
    cost = ProportionalCost(fee=0.0)
    w = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(cost(w), [0.0, 0.0])


def test_constant_weights_zero_turnover_cost():
    cost = ProportionalCost(fee=0.01)
    w = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    # only the initial position is charged
    assert np.allclose(cost(w), [0.01, 0.0, 0.0])


def test_parity_with_transaction_cost():
    cost = ProportionalCost(fee=0.002, slippage=0.001)
    w = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    assert np.allclose(cost(w), transaction_cost(w, fee=0.003))


def test_conforms_to_protocol():
    assert isinstance(ProportionalCost(), CostModel)


# -- MarketImpactCost -----------------------------------------------------


def test_impact_conforms_to_protocol():
    assert isinstance(MarketImpactCost(), CostModel)


def test_impact_linear_limit_equals_proportional():
    # exponent=1, impact=0 reduces to ProportionalCost(fee)
    w = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    lin = MarketImpactCost(fee=0.003, impact=0.0, exponent=1.0)
    prop = ProportionalCost(fee=0.003)
    assert np.allclose(lin(w), prop(w), atol=1e-12)


def test_impact_convex_in_turnover():
    # Doubling a single-step turnover more-than-doubles its impact cost.
    small = np.array([[0.0], [0.25], [0.25]])   # step-1 turnover 0.25
    big = np.array([[0.0], [0.5], [0.5]])       # step-1 turnover 0.5 (doubled)
    cost = MarketImpactCost(fee=0.0, impact=0.1, exponent=1.5)
    c_small = cost(small)[1]
    c_big = cost(big)[1]
    assert c_big > 2.0 * c_small


def test_impact_zero_turnover_zero_cost():
    w = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    cost = MarketImpactCost(fee=0.01, impact=0.1)
    # only the initial position is charged; steps 2-3 have zero turnover
    assert np.allclose(cost(w)[1:], [0.0, 0.0])


def test_impact_shape_and_first_step_charged():
    w = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    cost = MarketImpactCost(fee=0.001, impact=0.05)
    out = cost(w)
    assert out.shape == (3,)
    assert out[0] > 0.0   # initial position charged like ProportionalCost


def test_impact_bad_exponent_raises():
    with pytest.raises(ValueError, match="exponent"):
        MarketImpactCost(exponent=0.0)


def test_proportional_components_sum_to_total():
    cost = ProportionalCost(fee=0.002, slippage=0.001)
    w = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    comps = cost.components(w)
    assert set(comps) == {"transaction"}
    assert np.allclose(comps["transaction"], cost(w))


def test_impact_components_split_and_sum_to_total():
    cost = MarketImpactCost(fee=0.001, impact=0.05, exponent=1.5)
    w = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    comps = cost.components(w)
    assert set(comps) == {"transaction", "market_impact"}
    # The two components add up to the aggregate per-step cost.
    assert np.allclose(comps["transaction"] + comps["market_impact"], cost(w))
    # The convex impact term is strictly positive on a real trade.
    assert comps["market_impact"][0] > 0.0


def test_impact_lowers_net_equity_in_backtest():
    rng = np.random.default_rng(0)
    T = 300
    returns = rng.standard_normal((T, 3)) * 0.01
    pos = rng.standard_normal((T, 3))
    pos = pos / np.abs(pos).sum(axis=1, keepdims=True)   # normalized weights

    base = backtest(returns, pos, cost=ProportionalCost(fee=0.001))
    impacted = backtest(
        returns, pos, cost=MarketImpactCost(fee=0.001, impact=0.05),
    )
    # impact adds a strictly positive convex term -> higher total cost,
    # lower final equity.
    assert impacted.costs.sum() > base.costs.sum()
    assert impacted.equity[-1] < base.equity[-1]
