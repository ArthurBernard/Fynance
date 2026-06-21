#!/usr/bin/env python3
# coding: utf-8

""" Tests for regime-adaptive rolling windows. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.features.engineering import adaptive_roll, adaptive_volatility
from fynance.features.indicators import realized_volatility
from fynance.features.momentums import sma


def test_single_regime_constant_window_equals_fixed():
    X = np.arange(1.0, 21.0)
    regimes = np.zeros(X.size, dtype=int)
    out = adaptive_roll(X, sma, {0: 3}, regimes)
    assert np.allclose(out, np.asarray(sma(X, 3)).reshape(-1))


def test_output_length_and_known_values():
    X = np.arange(1.0, 7.0)
    regimes = np.array([0, 0, 0, 1, 1, 1])
    out = adaptive_roll(X, sma, {0: 1, 1: 3}, regimes)
    assert out.shape == (6,)
    # regime 0 -> sma window 1 (= X); regime 1 -> sma window 3
    assert np.allclose(out, [1.0, 2.0, 3.0, 3.0, 4.0, 5.0])


def test_missing_window_for_regime_raises():
    X = np.arange(1.0, 7.0)
    regimes = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="no entry for regime"):
        adaptive_roll(X, sma, {0: 2}, regimes)


def test_length_mismatch_raises():
    X = np.arange(1.0, 7.0)
    with pytest.raises(ValueError, match="length"):
        adaptive_roll(X, sma, {0: 2}, np.zeros(3, dtype=int))


def test_no_lookahead():
    rng = np.random.default_rng(0)
    X = 100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
    regimes = (np.arange(200) // 50) % 2
    full = adaptive_volatility(X, {0: 5, 1: 20}, regimes)
    t = 120
    truncated = adaptive_volatility(X[:t], {0: 5, 1: 20}, regimes[:t])
    assert np.allclose(full[:t], truncated, equal_nan=True)


def test_window_varies_with_regime():
    # Calm segment then turbulent segment; a short window in the turbulent regime
    # tracks the vol jump faster than the long calm-window baseline.
    rng = np.random.default_rng(1)
    n = 400
    calm = rng.standard_normal(n // 2) * 0.003
    turbulent = rng.standard_normal(n // 2) * 0.03
    log_ret = np.concatenate([calm, turbulent])
    X = 100 * np.exp(np.cumsum(log_ret))
    regimes = (np.arange(n) >= n // 2).astype(int)   # 0 calm, 1 turbulent

    # short window (5) in the turbulent regime, long (50) in calm
    adaptive = adaptive_volatility(X, {0: 50, 1: 5}, regimes)
    # baseline: the long window everywhere
    baseline = np.asarray(realized_volatility(X, w=50, period=252)).reshape(-1)

    # just after the regime switch, the adaptive estimate (short window) has
    # risen more than the slow baseline.
    probe = n // 2 + 10
    assert adaptive[probe] > baseline[probe]


def test_adaptive_volatility_matches_manual_roll():
    rng = np.random.default_rng(2)
    X = 100 * np.exp(np.cumsum(rng.standard_normal(120) * 0.01))
    regimes = (np.arange(120) // 40) % 2
    auto = adaptive_volatility(X, {0: 7, 1: 21}, regimes, period=252)
    manual = adaptive_roll(
        X, realized_volatility, {0: 7, 1: 21}, regimes, period=252,
    )
    assert np.allclose(auto, manual, equal_nan=True)
