#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.3 market-regime detection. """

import numpy as np

from fynance.features.regime import detect_regimes


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
