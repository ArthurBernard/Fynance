#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-01 16:51:38
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-15 16:39:37

# Built-in packages
import warnings

# Third party packages
import numpy as np
import pytest
from scipy.special import gammaln

# local packages
import fynance as fy
from fynance.models.econometric_models import (
    _arma_garch,
    _egarch,
    _gjr_garch,
    _loglik_normal,
    _loglik_t,
    _mean_abs_standardized_t,
    loglik_garch,
)


@pytest.fixture()
def set_variables():
    y = np.array([60, 100, 80, 120, 160, 80])
    params = np.array([0.2, 0.5, -0.3, 0.6, 0.1, 0.1])
    p, q, P, Q = 1, 1, 1, 1
    return y, params, p, q, P, Q


def test_get_parameters():
    # Use distinct orders (p=1, q=1, Q=2, P=1) and distinct parameter values so
    # that a Q<->P packing swap, or an argument passed out of signature order,
    # changes one of the asserted slices and fails. The signature is
    # ``get_parameters(params, p, q, Q, P, cons)`` and, with a constant, the
    # flat layout is [c, phi(p), theta(q), omega, alpha(Q), beta(P)].
    p, q, Q, P = 1, 1, 2, 1
    # c=0.5, phi=0.3, theta=-0.2, omega=0.1, alpha=[0.4, 0.05], beta=0.8
    params = np.array([0.5, 0.3, -0.2, 0.1, 0.4, 0.05, 0.8])

    phi, theta, alpha, beta, c, omega = fy.get_parameters(
        params, p, q, Q, P, cons=True
    )
    assert c == params[0]
    assert np.array_equal(phi, params[1:2])
    assert np.array_equal(theta, params[2:3])
    assert omega == params[3]
    assert np.array_equal(alpha, params[4:6])  # Q=2 -> two ARCH coefficients
    assert np.array_equal(beta, params[6:7])   # P=1 -> one GARCH coefficient

    # Without a constant the constant slot disappears and everything shifts left.
    params_nc = np.array([0.3, -0.2, 0.1, 0.4, 0.05, 0.8])
    phi, theta, alpha, beta, c, omega = fy.get_parameters(
        params_nc, p, q, Q, P, cons=False
    )
    assert c == 0.
    assert np.array_equal(phi, params_nc[0:1])
    assert np.array_equal(theta, params_nc[1:2])
    assert omega == params_nc[2]
    assert np.array_equal(alpha, params_nc[3:5])
    assert np.array_equal(beta, params_nc[5:6])


def test_ARMA_GARCH(set_variables):
    y, params, p, q, P, Q = set_variables
    u_est, h_est = fy.ARMA_GARCH(
        y, phi=params[1:2], theta=params[2:3], alpha=params[4:5],
        beta=params[5:6], c=params[0], omega=params[3], p=1, q=1, P=1, Q=1
    )
    u = np.zeros([y.size])
    h = np.zeros([y.size])
    u[0] = y[0] - params[0]
    h[0] = np.sqrt(params[3])
    assert u[0] == u_est[0]
    assert h[0] == h_est[0]
    for t in range(1, y.size):
        u[t] = y[t] - params[0] - y[t - 1] * params[1] - u[t - 1] * params[2]
        h[t] = np.sqrt(
            params[3] + params[4] * u[t - 1] ** 2 + params[5] * h[t - 1] ** 2
        )
        assert u[t] == u_est[t]
        assert h[t] == h_est[t]


def test_MA_recurrence():
    # MA(q): u[0] = y[0] - c ; u[t] = y[t] - c - sum_j theta[j] * u[t-1-j]
    y = np.array([60., 100., 80., 120., 160., 80.])
    theta, c = np.array([0.8]), 3.0
    u_est = np.asarray(fy.MA(y=y, theta=theta, c=c, q=1))
    u = np.zeros(y.size)
    u[0] = y[0] - c
    for t in range(1, y.size):
        u[t] = y[t] - c - theta[0] * u[t - 1]
    assert np.allclose(u_est, u)


def test_ARMA_recurrence():
    # ARMA(1,1): u[t] = y[t] - c - phi*y[t-1] - theta*u[t-1]
    y = np.array([60., 100., 80., 120., 160., 80.])
    phi, theta, c = np.array([0.5]), np.array([-0.3]), 0.2
    u_est = np.asarray(fy.ARMA(y=y, phi=phi, theta=theta, c=c, p=1, q=1))
    u = np.zeros(y.size)
    u[0] = y[0] - c
    for t in range(1, y.size):
        u[t] = y[t] - c - phi[0] * y[t - 1] - theta[0] * u[t - 1]
    assert np.allclose(u_est, u)


def test_ARMAX_GARCH_shapes_and_positive_vol(set_variables):
    # Property-level coverage: residuals/vol have length T, are finite, and the
    # conditional standard deviation stays strictly positive.
    y, params, p, q, P, Q = set_variables
    y = y.astype(float)
    x = np.zeros((y.size, 1))
    u, h = fy.ARMAX_GARCH(
        y, x, phi=params[1:2], psi=np.array([0.1]), theta=params[2:3],
        alpha=params[4:5], beta=params[5:6], c=params[0], omega=params[3],
        p=1, q=1, P=1, Q=1,
    )
    u, h = np.asarray(u), np.asarray(h)
    assert u.shape == (y.size,) and h.shape == (y.size,)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(h))
    assert np.all(h > 0)


def test_ARMA_accepts_list_like_MA():
    # ARMA must coerce a plain Python list for y like MA does, instead of
    # crashing on ``'list' object has no attribute 'size'``.
    y_list = [60., 100., 80., 120., 160., 80.]
    u_list = np.asarray(fy.ARMA(y_list, phi=[0.5], theta=[-0.3], c=0.2, p=1, q=1))
    u_arr = np.asarray(
        fy.ARMA(np.array(y_list), phi=np.array([0.5]),
                theta=np.array([-0.3]), c=0.2, p=1, q=1)
    )
    assert np.allclose(u_list, u_arr)


def test_ARMA_GARCH_accepts_list():
    y_list = [60., 100., 80., 120., 160., 80.]
    u_l, h_l = fy.ARMA_GARCH(
        y_list, phi=[0.5], theta=[-0.3], alpha=[0.1], beta=[0.1],
        c=0.2, omega=0.6, p=1, q=1, Q=1, P=1,
    )
    u_a, h_a = fy.ARMA_GARCH(
        np.array(y_list), phi=np.array([0.5]), theta=np.array([-0.3]),
        alpha=np.array([0.1]), beta=np.array([0.1]),
        c=0.2, omega=0.6, p=1, q=1, Q=1, P=1,
    )
    assert np.allclose(np.asarray(u_l), np.asarray(u_a))
    assert np.allclose(np.asarray(h_l), np.asarray(h_a))


def test_ARMAX_GARCH_accepts_list():
    y_list = [60., 100., 80., 120., 160., 80.]
    x = np.zeros((len(y_list), 1))
    u_l, h_l = fy.ARMAX_GARCH(
        y_list, x, phi=[0.5], psi=[0.1], theta=[-0.3], alpha=[0.1],
        beta=[0.1], c=0.2, omega=0.6, p=1, q=1, Q=1, P=1,
    )
    u_a, h_a = fy.ARMAX_GARCH(
        np.array(y_list), x, phi=np.array([0.5]), psi=np.array([0.1]),
        theta=np.array([-0.3]), alpha=np.array([0.1]), beta=np.array([0.1]),
        c=0.2, omega=0.6, p=1, q=1, Q=1, P=1,
    )
    assert np.allclose(np.asarray(u_l), np.asarray(u_a))
    assert np.allclose(np.asarray(h_l), np.asarray(h_a))


# =========================================================================== #
#                  GJR-GARCH / EGARCH kernels & log-likelihood                #
# =========================================================================== #


# --- Slow pure-Python references (independent of the numba kernels) -------- #


def _gjr_ref(y, omega, alpha, gamma, beta):
    """ Reference GJR-GARCH(1, 1) conditional std recursion. """
    y = np.asarray(y, dtype=np.float64)
    h = np.zeros(y.size)
    h[0] = np.sqrt(omega)
    for t in range(1, y.size):
        ind = 1.0 if y[t - 1] < 0.0 else 0.0
        var_t = omega + (alpha + gamma * ind) * y[t - 1] ** 2 + beta * h[t - 1] ** 2
        h[t] = np.sqrt(var_t)
    return h


def _egarch_ref(y, omega, alpha, gamma, beta, mean_abs_z):
    """ Reference EGARCH(1, 1) conditional std recursion. """
    y = np.asarray(y, dtype=np.float64)
    h = np.zeros(y.size)
    log_var = omega
    h[0] = np.exp(0.5 * log_var)
    for t in range(1, y.size):
        z = y[t - 1] / h[t - 1]
        log_var = omega + beta * log_var + alpha * (abs(z) - mean_abs_z) + gamma * z
        h[t] = np.exp(0.5 * log_var)
    return h


def _mean_abs_std_t_ref(nu):
    """ Reference E|z| for a unit-variance Student-t via scipy gammaln. """
    ratio = np.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0))
    return 2.0 * np.sqrt(nu - 2.0) * ratio / ((nu - 1.0) * np.sqrt(np.pi))


def _loglik_normal_ref(y, h):
    """ Reference Gaussian log-likelihood given conditional std ``h``. """
    return float(np.sum(-0.5 * np.log(2.0 * np.pi) - np.log(h) - 0.5 * (y / h) ** 2))


def _loglik_t_ref(y, h, nu):
    """ Reference standardized Student-t log-likelihood given ``h``. """
    const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) \
        - 0.5 * np.log(np.pi * (nu - 2.0))
    z2 = (y / h) ** 2
    return float(np.sum(const - np.log(h) - 0.5 * (nu + 1.0) * np.log(1.0 + z2 / (nu - 2.0))))


# --- Filter golden parity -------------------------------------------------- #


def test_gjr_garch_parity_reference():
    rng = np.random.default_rng(1234)
    y = rng.standard_normal(500)
    omega, alpha, gamma, beta = 0.05, 0.06, 0.08, 0.85
    h_kernel = _gjr_garch(y, omega, alpha, gamma, beta)
    h_ref = _gjr_ref(y, omega, alpha, gamma, beta)
    assert np.allclose(h_kernel, h_ref, rtol=1e-12, atol=0.0)


def test_egarch_parity_reference():
    rng = np.random.default_rng(5678)
    y = rng.standard_normal(500)
    omega, alpha, gamma, beta = -0.1, 0.12, -0.08, 0.9
    e_abs = np.sqrt(2.0 / np.pi)
    h_kernel = _egarch(y, omega, alpha, gamma, beta, e_abs)
    h_ref = _egarch_ref(y, omega, alpha, gamma, beta, e_abs)
    assert np.allclose(h_kernel, h_ref, rtol=1e-12, atol=0.0)


def test_gjr_gamma_zero_equals_vanilla_garch():
    # With gamma == 0 the GJR filter must reproduce the vanilla GARCH h path
    # produced by the existing _arma_garch kernel, bit for bit.
    rng = np.random.default_rng(3)
    y = rng.standard_normal(300)
    omega, alpha, beta = 0.05, 0.08, 0.88
    h_gjr = _gjr_garch(y, omega, alpha, 0.0, beta)
    zeros = np.array([0.0])
    _, h_ag = _arma_garch(
        y, zeros, zeros, np.array([alpha]), np.array([beta]),
        0.0, omega, 0, 0, 1, 1,
    )
    assert np.array_equal(h_gjr, h_ag)


def test_egarch_three_step_recursion():
    # Hand-computed 3-step EGARCH recursion.
    omega, alpha, gamma, beta = -0.1, 0.12, -0.08, 0.9
    e_abs = np.sqrt(2.0 / np.pi)
    y = np.array([0.5, -0.7, 0.3, 0.9])
    h = _egarch(y, omega, alpha, gamma, beta, e_abs)

    lv0 = omega
    h0 = np.exp(0.5 * lv0)
    z0 = y[0] / h0
    lv1 = omega + beta * lv0 + alpha * (abs(z0) - e_abs) + gamma * z0
    h1 = np.exp(0.5 * lv1)
    z1 = y[1] / h1
    lv2 = omega + beta * lv1 + alpha * (abs(z1) - e_abs) + gamma * z1
    h2 = np.exp(0.5 * lv2)

    assert np.isclose(h[0], h0, rtol=1e-12)
    assert np.isclose(h[1], h1, rtol=1e-12)
    assert np.isclose(h[2], h2, rtol=1e-12)


# --- Log-likelihood parity ------------------------------------------------- #


def test_loglik_garch_normal_matches_reference():
    rng = np.random.default_rng(11)
    y = rng.standard_normal(300)
    omega, alpha, beta = 0.04, 0.06, 0.88
    h = _gjr_ref(y, omega, alpha, 0.0, beta)
    ll_ref = _loglik_normal_ref(y, h)
    ll = loglik_garch(np.array([omega, alpha, beta]), y, 'garch', 'normal')
    assert np.isclose(ll, ll_ref, rtol=1e-12)


def test_loglik_gjr_normal_matches_reference():
    rng = np.random.default_rng(12)
    y = rng.standard_normal(300)
    omega, alpha, gamma, beta = 0.04, 0.06, 0.05, 0.85
    h = _gjr_ref(y, omega, alpha, gamma, beta)
    ll_ref = _loglik_normal_ref(y, h)
    ll = loglik_garch(np.array([omega, alpha, gamma, beta]), y, 'gjr', 'normal')
    assert np.isclose(ll, ll_ref, rtol=1e-12)


def test_loglik_egarch_t_matches_reference():
    rng = np.random.default_rng(13)
    y = rng.standard_normal(300)
    omega, alpha, gamma, beta, nu = -0.1, 0.1, -0.05, 0.9, 6.0
    e_abs = _mean_abs_std_t_ref(nu)
    h = _egarch_ref(y, omega, alpha, gamma, beta, e_abs)
    ll_ref = _loglik_t_ref(y, h, nu)
    ll = loglik_garch(
        np.array([omega, alpha, gamma, beta, nu]), y, 'egarch', 't',
    )
    assert np.isclose(ll, ll_ref, rtol=1e-12)


def test_loglik_t_approaches_normal_large_nu():
    # As nu -> inf the standardized Student-t density collapses onto the
    # normal, so both log-likelihoods must agree (atol scaled to T ~ 400).
    rng = np.random.default_rng(7)
    y = rng.standard_normal(400)
    params_n = np.array([0.05, 0.08, 0.85])
    ll_n = loglik_garch(params_n, y, 'garch', 'normal')
    ll_t = loglik_garch(np.append(params_n, 1e6), y, 'garch', 't')
    assert np.isclose(ll_t, ll_n, atol=1e-2)


def test_mean_abs_z_t_vs_numerical_integration():
    # The closed-form E|z| for a standardized Student-t must match a direct
    # numerical integration of |z| * f(z) on a fine grid.
    for nu in (3.0, 5.0, 8.0, 30.0):
        z = np.linspace(-3000.0, 3000.0, 6_000_001)
        const = np.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)) \
            / np.sqrt(np.pi * (nu - 2.0))
        f = const * (1.0 + z ** 2 / (nu - 2.0)) ** (-(nu + 1.0) / 2.0)
        num = np.trapezoid(np.abs(z) * f, z)
        assert np.isclose(_mean_abs_standardized_t(nu), num, rtol=1e-6)


def test_loglik_invalid_params_return_minus_inf_without_warnings():
    rng = np.random.default_rng(99)
    y = rng.standard_normal(200)
    cases = [
        (np.array([-0.1, 0.05, 0.9]), 'garch', 'normal'),        # omega <= 0
        (np.array([0.1, -0.05, 0.9]), 'garch', 'normal'),        # alpha < 0
        (np.array([0.1, 0.2, 0.9]), 'garch', 'normal'),          # non-stationary
        (np.array([0.1, 0.05, 0.9, 1.5]), 'garch', 't'),         # nu <= 2
        (np.array([0.1, 0.05, -0.3, 0.9]), 'gjr', 'normal'),     # alpha + gamma < 0
        (np.array([0.1, 0.05, 0.1, 0.95]), 'gjr', 'normal'),     # non-stationary
        (np.array([-0.1, 0.1, -0.05, 1.2]), 'egarch', 'normal'),  # |beta| >= 1
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for params, model, dist in cases:
            assert loglik_garch(params, y, model, dist) == -np.inf


def test_garch_kernels_are_numba_compiled():
    for fn in (
        _gjr_garch, _egarch, _mean_abs_standardized_t, _loglik_normal, _loglik_t,
    ):
        assert hasattr(fn, 'py_func')
