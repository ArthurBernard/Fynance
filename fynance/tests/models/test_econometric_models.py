#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-01 16:51:38
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-15 16:39:37

# Built-in packages

# Third party packages
import numpy as np
import pytest

# local packages
import fynance as fy


@pytest.fixture()
def set_variables():
    y = np.array([60, 100, 80, 120, 160, 80])
    params = np.array([0.2, 0.5, -0.3, 0.6, 0.1, 0.1])
    p, q, P, Q = 1, 1, 1, 1
    return y, params, p, q, P, Q


def test_get_parameters(set_variables):
    y, params, p, q, P, Q = set_variables
    phi, theta, alpha, beta, c, omega = fy.get_parameters(
        params, p, q, P, Q, cons=True
    )
    assert c == params[0]
    assert phi == params[1]
    assert theta == params[2]
    assert omega == params[3]
    assert alpha == params[4]
    assert beta == params[5]
    phi, theta, alpha, beta, c, omega = fy.get_parameters(
        params, p, q, P, Q, cons=False
    )
    assert phi == params[0]
    assert theta == params[1]
    assert omega == params[2]
    assert alpha == params[3]
    assert beta == params[4]


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
