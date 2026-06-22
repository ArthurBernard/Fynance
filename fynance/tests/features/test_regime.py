#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.3 market-regime detection. """

import numpy as np

from fynance.features.regime import _vol_order, detect_regimes, regime_features


def _two_regime_series(seed=0):
    rng = np.random.RandomState(seed)
    r = np.r_[rng.standard_normal(120) * 0.004, rng.standard_normal(120) * 0.03]
    return 100 * np.cumprod(1 + r)


def test_labels_shape_and_range():
    X = _two_regime_series()
    labels = detect_regimes(X, n_regimes=3, w=15)
    assert labels.shape == (240,)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_labels_ordered_by_volatility():
    X = _two_regime_series()
    labels = detect_regimes(X, n_regimes=2, w=15)
    # calm first half should get a lower (calmer) regime label than the volatile half
    assert labels[20:120].mean() < labels[140:].mean()


def test_deterministic_with_seed():
    X = _two_regime_series()
    a = detect_regimes(X, n_regimes=2, w=15, seed=42)
    b = detect_regimes(X, n_regimes=2, w=15, seed=42)
    assert np.array_equal(a, b)


def test_vol_order_empty_cluster_safe():
    # An empty cluster yields a nan mean; it must sort last (treated as +inf),
    # never corrupt the order of the populated clusters.
    vol = np.array([0.1, 0.5, 0.3, 0.2])
    labels = np.array([0, 2, 0, 2])  # cluster 1 is empty
    order = _vol_order(vol, labels, n_regimes=3)
    # Populated means: 0 -> 0.2, 2 -> 0.35; empty 1 -> +inf -> last.
    assert order.tolist() == [0, 2, 1]


def test_vol_order_matches_naive_when_all_populated():
    rng = np.random.RandomState(7)
    vol = rng.rand(50)
    labels = rng.randint(0, 3, 50)
    order = _vol_order(vol, labels, n_regimes=3)
    naive = np.argsort([vol[labels == k].mean() for k in range(3)])
    assert order.tolist() == naive.tolist()


def test_regime_features_warmup_row_is_zero():
    # Row 0 is the documented deterministic warmup: (vol, mean log-return) = (0, 0).
    X = _two_regime_series()
    f = regime_features(X, w=15)
    assert f[0, 0] == 0.0
    assert f[0, 1] == 0.0
