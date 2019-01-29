#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
import pytest

# Internal packages
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
        u[t] = y[t] - params[0] - y[t-1] * params[1] - u[t-1] * params[2]
        h[t] = np.sqrt(params[3] + params[4] * u[t-1] ** 2 + params[5] * h[t-1] ** 2)
        assert u[t] == u_est[t]
        assert h[t] == h_est[t]