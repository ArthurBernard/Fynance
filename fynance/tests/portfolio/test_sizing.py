#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.8 position sizing + transaction costs. """

import warnings

import numpy as np
import pytest

from fynance.features.indicators import realized_volatility
from fynance.portfolio.sizing import (
    book_vol_target,
    kelly_fraction,
    transaction_cost,
    vol_target,
)


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


def test_book_vol_target_reduces_to_vol_target():
    # Single asset, unit weight: the book IS the asset (up to the 100-base
    # rebasing, which does not change log-returns), so realized vol and
    # hence leverage match vol_target exactly.
    rng = np.random.default_rng(42)
    T = 300
    X = 100 * np.cumprod(1 + rng.standard_normal(T) * 0.01)
    W = np.ones((T, 1))

    lev_book = np.asarray(book_vol_target(W, X.reshape(-1, 1), target_vol=0.15, w=21))
    lev_single = np.asarray(vol_target(X, target_vol=0.15, w=21))

    assert np.allclose(lev_book, lev_single, rtol=1e-12)


def test_book_vol_target_no_lookahead():
    rng = np.random.default_rng(7)
    T, N = 400, 4
    X = 100 * np.cumprod(1 + rng.standard_normal((T, N)) * 0.01, axis=0)
    W = np.full((T, N), 1.0 / N)

    t0 = 200
    lev = np.asarray(book_vol_target(W, X))
    X2 = X.copy()
    X2[t0:] *= 1.5
    lev2 = np.asarray(book_vol_target(W, X2))

    assert np.array_equal(lev[:t0], lev2[:t0])
    assert not np.allclose(lev[t0:], lev2[t0:])


def test_book_vol_target_hits_target():
    rng = np.random.default_rng(123)
    T, N = 1000, 3
    sigma = 0.01
    r = rng.standard_normal((T, N)) * sigma
    r[0] = 0.0
    X = 100 * np.cumprod(1 + r, axis=0)
    W = np.full((T, N), 1.0 / N)
    target_vol = 0.15

    lev = np.asarray(book_vol_target(W, X, target_vol=target_vol))

    # Recompute the trailing book vol with the same convention as inside
    # book_vol_target to check lev * trailing_vol ~= target_vol.
    ret = np.zeros_like(X)
    ret[1:] = X[1:] / X[:-1] - 1.0
    rb = np.zeros(T)
    rb[1:] = np.sum(W[:-1] * ret[1:], axis=1)
    L = 100.0 * np.cumprod(1.0 + rb)
    trailing_vol = np.asarray(realized_volatility(L, w=21, period=252))

    post_warmup = slice(101, T)
    realized_target = np.median((lev * trailing_vol)[post_warmup])
    assert abs(realized_target - target_vol) / target_vol < 0.20


def test_book_vol_target_cap_and_zero_variance():
    rng = np.random.default_rng(99)
    T, N = 300, 2

    # Tiny-vol synthetic: leverage should saturate at max_leverage everywhere
    # past the warm-up window.
    tiny = rng.standard_normal((T, N)) * 1e-6
    tiny[0] = 0.0
    X_tiny = 100 * np.cumprod(1 + tiny, axis=0)
    W = np.full((T, N), 1.0 / N)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lev_tiny = np.asarray(
            book_vol_target(W, X_tiny, target_vol=0.15, w=21, max_leverage=5.0)
        )
    assert np.allclose(lev_tiny[50:], 5.0)

    # Zero-variance stretch: constant prices -> zero leverage, no warnings.
    X_flat = np.full((T, N), 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lev_flat = np.asarray(
            book_vol_target(W, X_flat, target_vol=0.15, w=21, max_leverage=5.0)
        )
    assert np.all(lev_flat == 0.0)


def test_book_vol_target_shape_mismatch():
    T = 50
    X = 100 * np.cumprod(1 + np.random.default_rng(0).standard_normal((T, 4)) * 0.01, axis=0)

    with pytest.raises(ValueError):
        book_vol_target(np.ones((T, 3)), X)

    with pytest.raises(ValueError):
        book_vol_target(np.ones((T - 1, 4)), X)
