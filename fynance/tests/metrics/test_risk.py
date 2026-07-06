#!/usr/bin/env python3
# coding: utf-8

""" Tests for the tail-risk metrics (VaR, CVaR, CDaR, tail dependence). """

# Third-party packages
import numpy as np
import pytest
from scipy.stats import norm

# Local packages
from fynance.metrics import METRICS, cdar, cvar, summary, tail_dependence, var
from fynance.metrics.risk import _cf_quantile, _cf_tail_constants, roll_cvar, roll_var

# --------------------------------------------------------------------------- #
# Gaussian closed forms
# --------------------------------------------------------------------------- #


def _sigma_scaled_normal(sigma, T=5000, seed=7):
    # Exactly demeaned and rescaled so the *sample* mean/std equal 0/sigma:
    # removes sampling noise from the closed-form comparison below, so the
    # only source of discrepancy is the rounding of the hardcoded constants.
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(T)
    r -= r.mean()
    r *= sigma / r.std()
    X = np.empty(T + 1)
    X[0] = 100.
    X[1:] = 100. * np.cumprod(1. + r)

    return X


def test_var_gaussian_closed_form_matches_hardcoded_constant():
    sigma = 0.01
    X = _sigma_scaled_normal(sigma)
    assert var(X, alpha=0.05, method='gaussian') == pytest.approx(sigma * 1.6449, rel=1e-3)


def test_cvar_gaussian_closed_form_matches_hardcoded_constant():
    sigma = 0.01
    X = _sigma_scaled_normal(sigma)
    assert cvar(X, alpha=0.05, method='gaussian') == pytest.approx(sigma * 2.0627, rel=1e-3)


# --------------------------------------------------------------------------- #
# Historical method, hand-built series -> exact empirical quantile / tail mean
# --------------------------------------------------------------------------- #


def _hand_built_returns_and_prices():
    # 100 clean, distinct per-period returns from -0.50 to +0.49 (step 0.01):
    # sorted ascending by construction, so the k-th smallest is a plain index.
    R = 0.01 * (np.arange(100) - 50)
    X = np.empty(101)
    X[0] = 100.
    for i in range(100):
        X[i + 1] = X[i] * (1. + R[i])

    return R, X


def test_var_historical_hand_built_exact_quantile():
    R, X = _hand_built_returns_and_prices()
    # alpha=0.05, n=100 -> k = floor(5.0) = 5 -> the 5th smallest observed return.
    expected_q = np.sort(R)[4]
    assert var(X, alpha=0.05, method='historical') == pytest.approx(-expected_q, abs=1e-9)


def test_cvar_historical_hand_built_exact_tail_mean():
    R, X = _hand_built_returns_and_prices()
    expected_tail_mean = np.mean(np.sort(R)[:5])
    assert cvar(X, alpha=0.05, method='historical') == pytest.approx(-expected_tail_mean, abs=1e-9)


def test_cvar_historical_at_least_var_historical():
    _, X = _hand_built_returns_and_prices()
    assert cvar(X, alpha=0.05, method='historical') >= var(X, alpha=0.05, method='historical')


# --------------------------------------------------------------------------- #
# Cornish-Fisher reduces to the Gaussian formula when skew = kurt_excess = 0
# --------------------------------------------------------------------------- #


def test_cf_quantile_reduces_to_gaussian_at_zero_moments():
    z = norm.ppf(0.05)
    assert _cf_quantile(z, 0., 0.) == pytest.approx(z)


def test_cf_tail_mean_reduces_to_gaussian_closed_form_at_zero_moments():
    alpha = 0.05
    c1, c2, c3, c4 = _cf_tail_constants(alpha)
    z = norm.ppf(alpha)
    gaussian_raw_tail_mean = -norm.pdf(z) / alpha  # mu=0, sigma=1 tail mean
    cf_raw_tail_mean = c1 + c2 * 0. + c3 * 0. + c4 * 0. ** 2
    assert cf_raw_tail_mean == pytest.approx(gaussian_raw_tail_mean, rel=1e-3)


# --------------------------------------------------------------------------- #
# CDaR, hand-built drawdown path
# --------------------------------------------------------------------------- #


def test_cdar_hand_built_known_drawdowns():
    # Drawdown path (percentage): [0, .1, .05, .2, 0, 40/110, 0, .5, 0].
    X = np.array([100., 90., 95., 80., 110., 70., 120., 60., 130.])
    # alpha=0.3, n=9 -> k = floor(2.7) = 2 -> mean of the 2 worst drawdowns.
    expected = (0.5 + 40. / 110.) / 2.
    assert cdar(X, alpha=0.3) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Rolling variants: causality (no future leakage) and NaN head
# --------------------------------------------------------------------------- #


def _rolling_price_curve(T=300, seed=3):
    rng = np.random.default_rng(seed)
    X = np.empty(T + 1)
    X[0] = 100.
    X[1:] = 100. * np.cumprod(1. + rng.normal(0., 0.01, T))

    return X


@pytest.mark.parametrize('method', ['historical', 'gaussian', 'cornish_fisher'])
@pytest.mark.parametrize('func', [roll_var, roll_cvar])
def test_roll_var_cvar_nan_head_and_causality(func, method):
    X = _rolling_price_curve()
    w = 50
    out = func(X, alpha=0.05, w=w, method=method)

    assert np.isnan(out[:w]).all()
    assert not np.isnan(out[w:]).any()

    t0 = 150
    X_perturbed = X.copy()
    X_perturbed[t0:] *= 1.5  # corrupt the future only
    out_perturbed = func(X_perturbed, alpha=0.05, w=w, method=method)
    assert np.allclose(out[:t0], out_perturbed[:t0], equal_nan=True), \
        f"{func.__name__}({method}) leaks future data"


# --------------------------------------------------------------------------- #
# tail_dependence
# --------------------------------------------------------------------------- #


def test_tail_dependence_comonotonic_pair_is_near_one():
    rng = np.random.default_rng(0)
    comonotone = rng.standard_normal(2000)
    R = np.stack([comonotone, comonotone], axis=1)
    lam = tail_dependence(R, q=0.05)
    assert lam[0, 1] > 0.95
    assert lam[1, 0] > 0.95


def test_tail_dependence_independent_large_t_is_near_q():
    rng = np.random.default_rng(11)
    T = 20000
    R = np.stack([rng.standard_normal(T), rng.standard_normal(T)], axis=1)
    lam = tail_dependence(R, q=0.05)
    assert abs(lam[0, 1] - 0.05) < 0.03


def test_tail_dependence_diagonal_and_symmetry():
    rng = np.random.default_rng(5)
    R = rng.standard_normal((500, 4))
    lam = tail_dependence(R, q=0.05)
    assert np.allclose(np.diag(lam), 1.)
    assert np.allclose(lam, lam.T)


# --------------------------------------------------------------------------- #
# Registry: var / cvar / cdar are single-curve scalar metrics, mirroring how
# the METRICS registry drives `summary` (see test_summary.py).
# --------------------------------------------------------------------------- #


def test_var_cvar_cdar_registered_in_metrics():
    assert {'var', 'cvar', 'cdar'} <= set(METRICS)
    eq = np.array([100., 101., 103., 102., 105., 107.])
    assert np.isclose(float(METRICS['var'](eq)), float(var(eq)))
    assert np.isclose(float(METRICS['cvar'](eq)), float(cvar(eq)))
    assert np.isclose(float(METRICS['cdar'](eq)), float(cdar(eq)))


def test_var_cvar_cdar_appear_in_summary_output():
    eq = np.array([100., 101., 103., 102., 105., 107.])
    s = summary(eq)
    assert {'var', 'cvar', 'cdar'} <= set(s)
    assert np.isclose(s['var'], float(var(eq)))
    assert np.isclose(s['cvar'], float(cvar(eq)))
    assert np.isclose(s['cdar'], float(cdar(eq)))


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_var_rejects_unknown_method():
    X = np.array([100., 101., 99., 102.])
    with pytest.raises(ValueError):
        var(X, method='not-a-method')


def test_var_rejects_invalid_alpha():
    X = np.array([100., 101., 99., 102.])
    with pytest.raises(ValueError):
        var(X, alpha=0.)
    with pytest.raises(ValueError):
        var(X, alpha=1.)
