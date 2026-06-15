#!/usr/bin/env python3
# coding: utf-8

""" Tests for transaction cost models. """

# Third-party packages
import numpy as np

# Local packages
from fynance.algorithms.sizing import transaction_cost
from fynance.backtest.cost import ProportionalCost
from fynance.core import CostModel


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
