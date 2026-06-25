#!/usr/bin/env python3
# coding: utf-8

""" Tests for the vectorized backtest engine. """

# Third-party packages
import numpy as np

# Local packages
from fynance.backtest import BacktestResult, backtest
from fynance.backtest.cost import MarketImpactCost, ProportionalCost


def test_buy_and_hold_matches_price_path():
    prices = np.array([100.0, 110.0, 99.0, 108.9])
    res = backtest(prices, np.ones(prices.size), returns_input=False, shift=False)
    # full long: equity tracks the (normalized) price path
    expected = prices[1:] / prices[0]
    assert np.allclose(res.equity, expected)


def test_flat_position_flat_equity():
    returns = np.array([0.01, -0.02, 0.03])
    res = backtest(returns, np.zeros(returns.size))
    assert np.allclose(res.equity, 1.0)


def test_causal_shift():
    returns = np.array([0.10, 0.20, -0.30])
    positions = np.array([1.0, 1.0, 1.0])
    res = backtest(returns, positions, shift=True)
    # first period earns nothing (no prior position)
    assert res.gross_returns[0] == 0.0
    assert np.isclose(res.gross_returns[1], 0.20)
    assert np.isclose(res.gross_returns[2], -0.30)


def test_no_lookahead():
    returns = np.array([0.01, 0.02, -0.03, 0.04, 0.05])
    positions = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    base = backtest(returns, positions, shift=True).equity.copy()
    # perturb the LAST return only -> equity up to the penultimate step unchanged
    perturbed_returns = returns.copy()
    perturbed_returns[-1] = 99.0
    pert = backtest(perturbed_returns, positions, shift=True).equity
    assert np.allclose(base[:-1], pert[:-1])


def test_costs_reduce_net_returns():
    returns = np.array([0.01, 0.02, 0.03])
    positions = np.array([1.0, 0.0, 1.0])
    cost = ProportionalCost(fee=0.01)
    res = backtest(returns, positions, cost=cost, shift=True)
    no_cost = backtest(returns, positions, shift=True)
    assert np.all(res.returns <= no_cost.returns + 1e-12)
    assert np.allclose(no_cost.returns - res.returns, cost(positions))


def test_two_asset_book():
    returns = np.array([[0.01, 0.02], [0.03, -0.01], [0.0, 0.05]])
    weights = np.array([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
    res = backtest(returns, weights, shift=True)
    # manual: gross_t = w_{t-1} . r_t
    assert res.gross_returns[0] == 0.0
    assert np.isclose(res.gross_returns[1], 0.5 * 0.03 + 0.5 * -0.01)
    assert np.isclose(res.gross_returns[2], 0.5 * 0.0 + 0.5 * 0.05)


def test_returns_result_type():
    res = backtest(np.array([0.01, 0.02]), np.array([1.0, 1.0]))
    assert isinstance(res, BacktestResult)


def test_prices_input_cost_turnover_timing():
    # On a prices input the engine drops the first position (it predates the
    # first return). The cost model must still see the full book so the initial
    # trade is charged once and a *constant* book then incurs no further cost.
    prices = np.array([100.0, 110.0, 99.0, 108.9])
    positions = np.ones(prices.size)          # constant full long
    cost = ProportionalCost(fee=0.01)
    res = backtest(prices, positions, cost=cost, returns_input=False, shift=True)
    # 3 returns; only the initial entry is charged, no spurious re-entry.
    assert res.costs.shape == res.returns.shape
    assert np.allclose(res.costs, [0.01, 0.0, 0.0])


def test_prices_input_cost_matches_returns_input():
    # The prices path and the equivalent returns path must agree on cost timing.
    prices = np.array([100.0, 110.0, 99.0, 108.9, 120.0])
    positions = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
    cost = ProportionalCost(fee=0.02)

    res_p = backtest(prices, positions, cost=cost,
                     returns_input=False, shift=True)

    returns = prices[1:] / prices[:-1] - 1.0
    # Equivalent returns-input call uses the same surviving positions and the
    # full book (including the dropped first entry) as the cost reference.
    res_r = backtest(returns, positions[1:], cost=cost,
                     returns_input=True, shift=True)
    expected_costs = cost(positions[:-1])      # turnover into earning positions
    assert np.allclose(res_p.costs, expected_costs)
    assert np.allclose(res_p.gross_returns, res_r.gross_returns)


def test_prices_input_cost_preserves_no_lookahead():
    # Re-assert causality on the prices+cost path (no regression of the shift).
    prices = np.array([100.0, 101.0, 102.0, 99.0, 105.0, 110.0])
    positions = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    cost = ProportionalCost(fee=0.01)
    base = backtest(prices, positions, cost=cost,
                    returns_input=False, shift=True).equity.copy()
    pert = prices.copy()
    pert[-1] = 999.0                            # perturb the future only
    res = backtest(pert, positions, cost=cost,
                   returns_input=False, shift=True).equity
    assert np.allclose(base[:-1], res[:-1])


def test_book_attribution_sums_to_gross():
    # A 2-D position book yields per-asset gross contributions that sum to the
    # aggregated book gross return.
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, (50, 3))
    positions = rng.choice([-1.0, 1.0], size=(50, 3))
    res = backtest(returns, positions, shift=True)
    assert res.asset_gross_returns is not None
    assert res.asset_gross_returns.shape == (50, 3)
    assert np.allclose(res.asset_gross_returns.sum(axis=1), res.gross_returns)


def test_single_asset_has_no_attribution():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, 50)
    res = backtest(returns, np.ones(50), shift=True)
    assert res.asset_gross_returns is None
    assert res.gross_returns.ndim == 1


def test_cost_components_captured_and_sum_to_total():
    returns = np.array([0.01, 0.02, 0.03, -0.01])
    positions = np.array([1.0, 0.0, 1.0, -1.0])
    cost = MarketImpactCost(fee=0.001, impact=0.05, exponent=1.5)
    res = backtest(returns, positions, cost=cost, shift=True)
    assert res.cost_components is not None
    assert set(res.cost_components) == {"transaction", "market_impact"}
    stacked = sum(res.cost_components.values())
    assert np.allclose(stacked, res.costs)


def test_cost_components_none_without_cost_model():
    # No cost model -> no breakdown to carry.
    res = backtest(np.array([0.01, 0.02]), np.array([1.0, 1.0]))
    assert res.cost_components is None
