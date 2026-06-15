#!/usr/bin/env python3
# coding: utf-8

""" Tests for the vectorized backtest engine. """

# Third-party packages
import numpy as np

# Local packages
from fynance.backtest import BacktestResult, backtest
from fynance.backtest.cost import ProportionalCost


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
