#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.8 position sizing + transaction costs. """

import numpy as np

from fynance.algorithms.sizing import kelly_fraction, transaction_cost, vol_target


def test_kelly_fraction_formula():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.02])
    assert np.isclose(kelly_fraction(r, fraction=0.5), 0.5 * r.mean() / r.var(ddof=0))


def test_kelly_zero_variance():
    assert kelly_fraction(np.zeros(10)) == 0.0


def test_vol_target_shape_and_cap():
    rng = np.random.RandomState(0)
    X = 100 * np.cumprod(1 + rng.standard_normal(120) * 0.01)
    lev = np.asarray(vol_target(X, target_vol=0.15, w=21, max_leverage=3.0))
    assert lev.shape == (120,)
    assert lev.min() >= 0.0 and lev.max() <= 3.0


def test_vol_target_no_lookahead():
    rng = np.random.RandomState(1)
    X = 100 * np.cumprod(1 + rng.standard_normal(100) * 0.01)
    t = 60
    base = np.asarray(vol_target(X, w=21))
    X2 = X.copy()
    X2[t:] *= 1.5
    pert = np.asarray(vol_target(X2, w=21))
    assert np.allclose(base[:t], pert[:t])


def test_transaction_cost_turnover():
    w = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]])
    cost = transaction_cost(w, fee=0.01)
    # step0: |1|+|0| = 1 -> 0.01 ; step1: |0.5|+|0.5| = 1 -> 0.01 ; step2: 0
    assert np.allclose(cost, [0.01, 0.01, 0.0])


def test_transaction_cost_1d():
    w = np.array([0.0, 1.0, 1.0, -1.0])
    cost = transaction_cost(w, fee=0.001)
    assert np.allclose(cost, [0.0, 0.001, 0.0, 0.002])
