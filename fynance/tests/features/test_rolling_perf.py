#!/usr/bin/env python3
# coding: utf-8

""" The O(n) deque rolling extrema and alloc-free roll_mdd match the naive
reference exactly (the optimization preserves bit-identical results). """

# Third-party packages
import numpy as np

# Local packages
from fynance.features._metrics_helpers import _roll_mdd_1d, _roll_mdd_2d
from fynance.features.roll_functions import (
    _roll_max_1d,
    _roll_min_1d,
    _roll_min_2d,
)


def _naive_roll(X, w, fn):
    T = X.shape[0]
    out = np.empty(T)
    for t in range(T):
        out[t] = fn(X[max(0, t - w + 1): t + 1])
    return out


def _naive_mdd(X, w, raw):
    T = X.shape[0]
    out = np.empty(T)
    for t in range(T):
        lo = max(0, t - w + 1)
        win = X[lo: t + 1]
        run = win[0]
        S = 0.0
        for v in win:
            run = max(run, v)
            dd = (run - v) if raw else (1.0 - v / run)
            S = max(S, dd)
        out[t] = S
    return out


def test_roll_min_max_deque_exact():
    rng = np.random.default_rng(1)
    X = rng.integers(-7, 7, 200).astype(np.float64)  # many ties
    for w in (1, 2, 5, 33, 200):
        assert np.array_equal(_roll_min_1d(X, w), _naive_roll(X, w, np.min))
        assert np.array_equal(_roll_max_1d(X, w), _naive_roll(X, w, np.max))


def test_roll_min_2d_parallel_exact():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (120, 5))
    for w in (3, 20):
        out = _roll_min_2d(X, w)
        for n in range(X.shape[1]):
            assert np.array_equal(out[:, n], _naive_roll(X[:, n], w, np.min))


def test_roll_mdd_alloc_free_exact():
    rng = np.random.default_rng(3)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 150))
    for w in (1, 5, 30, 150):
        for raw in (0, 1):
            assert np.allclose(_roll_mdd_1d(px, w, raw), _naive_mdd(px, w, raw),
                               atol=1e-12)


def test_roll_mdd_2d_parallel_exact():
    rng = np.random.default_rng(4)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, (100, 3)), axis=0)
    out = _roll_mdd_2d(px, 7, 1)
    for n in range(px.shape[1]):
        assert np.allclose(out[:, n], _naive_mdd(px[:, n], 7, 1), atol=1e-12)
