#!/usr/bin/env python3
# coding: utf-8

""" Tests for the causal GARCH(1,1) volatility feature. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.features import garch_volatility


def _simulate_garch(T=1500, omega=1e-6, alpha=0.08, beta=0.9, seed=0):
    """ Simulate a GARCH(1,1) return path; return (returns, true_sigma). """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T)
    r = np.zeros(T)
    s2 = np.zeros(T)
    s2[0] = omega / (1.0 - alpha - beta)

    for t in range(1, T):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * eps[t]

    return r, np.sqrt(s2)


def test_shape_warmup_and_nonnegative():
    r, _ = _simulate_garch()
    sigma = garch_volatility(r, min_train=500)
    assert sigma.shape == r.shape
    assert np.all(np.isnan(sigma[:500]))
    valid = sigma[500:]
    assert np.all(valid >= 0.0)
    assert not np.any(np.isnan(valid))


def test_recovers_latent_volatility():
    r, true_sigma = _simulate_garch()
    sigma = garch_volatility(r, min_train=500)
    mask = ~np.isnan(sigma)
    corr = np.corrcoef(sigma[mask], true_sigma[mask])[0, 1]
    assert corr > 0.8


def test_no_lookahead_bitwise():
    r, _ = _simulate_garch()
    t = 900
    truncated = garch_volatility(r[:t], min_train=500)[-1]
    extended = garch_volatility(r[:t + 50], min_train=500)[t - 1]
    assert truncated == extended


def test_refit_is_causal_and_changes_results():
    r, _ = _simulate_garch()
    once = garch_volatility(r, min_train=500)
    refit = garch_volatility(r, min_train=500, refit=200)
    # refit produces a (generally) different series ...
    assert not np.allclose(once[500:], refit[500:])
    # ... but stays causal at EVERY index, not only on refit checkpoints: within
    # the block [start, end) the value sigma_t is filtered with params fit on
    # r[:start] (start <= t), so it never sees r[t:]. A value is therefore
    # unchanged when the series is extended, regardless of where we truncate.
    t = 850   # mid-block (500 + 2*200 = 900 is the next checkpoint)
    a = garch_volatility(r[:t], min_train=500, refit=200)[-1]
    b = garch_volatility(r[:t + 50], min_train=500, refit=200)[t - 1]
    assert a == b


def test_per_index_causality_sweep_with_refit():
    # The strong causality guarantee: sigma_t is F_{t-1}-measurable at *every* t,
    # including between refit checkpoints. Truncating the series just after t
    # leaves sigma_t bit-identical -- no future leakage anywhere.
    r, _ = _simulate_garch(T=900)
    full = garch_volatility(r, min_train=500, refit=150)
    for t in range(500, r.size):
        trunc = garch_volatility(r[:t + 1], min_train=500, refit=150)
        assert trunc[t] == full[t], t


def test_bad_min_train_raises():
    r, _ = _simulate_garch(T=100)
    with pytest.raises(ValueError, match="min_train"):
        garch_volatility(r, min_train=100)
    with pytest.raises(ValueError, match="min_train"):
        garch_volatility(r, min_train=1)
