#!/usr/bin/env python3
# coding: utf-8

""" Tests for transaction cost models. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.backtest.cost import (
    CompositeCost,
    HoldingCost,
    MarketImpactCost,
    ProportionalCost,
)
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


# -- HoldingCost ------------------------------------------------------------


def test_holding_conforms_to_protocol():
    assert isinstance(HoldingCost(), CostModel)


def test_holding_long_only_unlevered_zero_borrow_financing_cash_credit():
    # Gross exposure never exceeds 1 and weights are never negative: borrow
    # and financing are zero on every bar, while idle cash earns a credit
    # (a non-positive cost) proportional to the unused capital 1 - gross.
    w = np.array([[0.5, 0.0], [0.3, 0.3], [0.0, 0.0]])
    period = 250
    cost = HoldingCost(borrow=0.02, financing=0.01, cash_rate=0.01, period=period)
    gross = np.abs(w).sum(axis=1)
    expected_cash = -(0.01 / period) * (1.0 - gross)
    assert np.allclose(cost(w), expected_cash, rtol=1e-12)
    comps = cost.components(w)
    assert np.allclose(comps["borrow"], [0.0, 0.0, 0.0])
    assert np.allclose(comps["financing"], [0.0, 0.0, 0.0])
    assert np.all(comps["cash"] <= 0.0)


def test_holding_short_book_charges_borrow_only():
    # A fully (and constantly) short, unit-gross book isolates the borrow
    # term: financing (gross never exceeds 1) and the cash credit (gross is
    # never below 1) both stay at zero on every bar.
    w = np.array([[-1.0], [-1.0], [-1.0]])
    period = 250
    cost = HoldingCost(borrow=0.02, financing=0.01, cash_rate=0.01, period=period)
    expected = np.full(3, 0.02 / period)
    assert np.allclose(cost(w), expected, rtol=1e-12)
    comps = cost.components(w)
    assert np.allclose(comps["borrow"], expected, rtol=1e-12)
    assert np.allclose(comps["financing"], [0.0, 0.0, 0.0])
    assert np.allclose(comps["cash"], [0.0, 0.0, 0.0])


def test_holding_leverage_two_charges_financing_on_excess_only():
    # A constant 2x-levered, long-only book isolates the financing term: it
    # is charged on the gross exposure in excess of 1 (i.e. 1.0 here), while
    # borrow (no shorts) and the cash credit (gross exceeds 1) are zero.
    w = np.array([[2.0], [2.0], [2.0]])
    period = 250
    cost = HoldingCost(borrow=0.02, financing=0.01, cash_rate=0.01, period=period)
    expected = np.full(3, 0.01 / period)
    assert np.allclose(cost(w), expected, rtol=1e-12)
    comps = cost.components(w)
    assert np.allclose(comps["financing"], expected, rtol=1e-12)
    assert np.allclose(comps["borrow"], [0.0, 0.0, 0.0])
    assert np.allclose(comps["cash"], [0.0, 0.0, 0.0])


def test_holding_zero_rates_zero_cost():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((3, 4))
    cost = HoldingCost(borrow=0.0, financing=0.0, cash_rate=0.0, period=252)
    assert np.allclose(cost(w), [0.0, 0.0, 0.0])


def test_holding_components_sum_to_total():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((50, 3)) * 1.5   # mixed long/short/leveraged book
    cost = HoldingCost(borrow=0.03, financing=0.015, cash_rate=0.01, period=252)
    comps = cost.components(w)
    assert set(comps) == {"borrow", "financing", "cash"}
    total = comps["borrow"] + comps["financing"] + comps["cash"]
    assert np.allclose(total, cost(w), rtol=1e-12)
    assert np.all(comps["cash"] <= 0.0)


# -- CompositeCost ------------------------------------------------------------


def test_composite_conforms_to_protocol():
    assert isinstance(CompositeCost([ProportionalCost()]), CostModel)


def test_composite_equals_sum_of_parts_through_backtest():
    rng = np.random.default_rng(3)
    T, N = 1000, 5
    returns = rng.standard_normal((T, N)) * 0.01
    pos = rng.standard_normal((T, N))
    pos = pos / np.abs(pos).sum(axis=1, keepdims=True) * 2.0   # gross ~= 2

    prop = ProportionalCost(fee=0.001)
    holding = HoldingCost(borrow=0.02, financing=0.01)

    result_prop = backtest(returns, pos, cost=prop)
    result_holding = backtest(returns, pos, cost=holding)
    result_composite = backtest(returns, pos, cost=CompositeCost([prop, holding]))

    manual_costs = result_prop.costs + result_holding.costs
    assert np.allclose(result_composite.costs, manual_costs, rtol=1e-12)

    manual_equity = np.cumprod(1.0 + result_composite.gross_returns - manual_costs)
    assert np.allclose(result_composite.equity, manual_equity, rtol=1e-10)


def test_composite_components_sum_to_total():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((20, 3))
    models = [ProportionalCost(fee=0.001), HoldingCost(borrow=0.02, financing=0.01)]
    cost = CompositeCost(models)
    comps = cost.components(w)
    total = sum(comps.values())
    assert np.allclose(total, cost(w), rtol=1e-12)


def test_composite_components_prefix_duplicate_keys():
    # Two ProportionalCost models both emit a "transaction" component: the
    # later one is disambiguated by prefixing its class name.
    w = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    cost = CompositeCost([ProportionalCost(fee=0.001), ProportionalCost(fee=0.002)])
    comps = cost.components(w)
    assert set(comps) == {"transaction", "ProportionalCost.transaction"}
    assert np.allclose(comps["transaction"] + comps["ProportionalCost.transaction"], cost(w))


def test_composite_components_merges_holding_terms():
    w = np.array([[1.0, 0.0], [-0.5, 0.5], [2.0, 0.0]])
    cost = CompositeCost(
        [ProportionalCost(fee=0.001), HoldingCost(borrow=0.02, financing=0.01)]
    )
    comps = cost.components(w)
    assert set(comps) == {"transaction", "borrow", "financing", "cash"}
