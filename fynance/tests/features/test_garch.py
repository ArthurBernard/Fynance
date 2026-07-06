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


# --------------------------------------------------------------------------- #
#                    model= / dist= passthrough (additive)                    #
# --------------------------------------------------------------------------- #

# Golden values captured from the UNMODIFIED develop implementation (before the
# model= / dist= passthrough was added), on ``_simulate_garch()`` (the default
# DGP). The default path must remain bit-for-bit identical.
_GOLDEN_DEFAULT = {
    500: 0.008607834883228261,
    501: 0.008713191698838677,
    750: 0.008089790529001382,
    999: 0.008529123281159049,
    1250: 0.008801664065656356,
    1499: 0.007836187008735469,
}
_GOLDEN_REFIT_200 = {
    500: 0.008607834883228261,
    501: 0.008713191698838677,
    750: 0.007586396186527904,
    999: 0.008043871085146348,
    1250: 0.053165718827182365,
    1499: 0.4263395993181483,
}


def test_default_path_bitwise_golden():
    # The default (garch / normal) estimation path is untouched: it reproduces
    # the golden captured from develop, bit-for-bit, with and without refit.
    r, _ = _simulate_garch()
    once = garch_volatility(r, min_train=500)
    refit = garch_volatility(r, min_train=500, refit=200)
    for i, v in _GOLDEN_DEFAULT.items():
        assert once[i] == v, i
    for i, v in _GOLDEN_REFIT_200.items():
        assert refit[i] == v, i


def test_default_kwargs_match_positional_default():
    # Passing the defaults explicitly hits the same path as omitting them.
    r, _ = _simulate_garch()
    base = garch_volatility(r, min_train=500)
    explicit = garch_volatility(r, min_train=500, model='garch', dist='normal')
    assert np.array_equal(base, explicit, equal_nan=True)


@pytest.mark.parametrize('model,dist', [('gjr', 't'), ('egarch', 'normal')])
def test_non_default_model_dist_shape_and_warmup(model, dist):
    r, _ = _simulate_garch(T=900)
    sigma = garch_volatility(r, min_train=500, model=model, dist=dist)
    assert sigma.shape == r.shape
    assert np.all(np.isnan(sigma[:500]))
    valid = sigma[500:]
    assert np.all(valid >= 0.0) and not np.any(np.isnan(valid))


@pytest.mark.parametrize('model,dist', [('gjr', 't'), ('egarch', 'normal')])
def test_non_default_path_is_causal_truncation(model, dist):
    # sigma_t is F_{t-1}-measurable for the gjr / egarch feature paths too:
    # extending the series with future observations leaves sigma_t bit-identical
    # (truncation-based no-lookahead check, as for the default path).
    r, _ = _simulate_garch(T=900)
    t = 800
    truncated = garch_volatility(
        r[:t], min_train=500, model=model, dist=dist,
    )[-1]
    extended = garch_volatility(
        r[:t + 50], min_train=500, model=model, dist=dist,
    )[t - 1]

    assert truncated == extended
