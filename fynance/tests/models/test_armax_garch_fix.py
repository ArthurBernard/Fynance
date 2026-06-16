#!/usr/bin/env python3
# coding: utf-8

""" Regression test for the ARMAX_GARCH psi/theta argument-order bug.

Previously the public wrapper passed (phi, theta, psi) to the kernel whose
signature is (phi, psi, theta), so the external-regressor coefficients (psi) and
the MA coefficients (theta) were applied to the wrong terms.
"""

# Third-party packages
import numpy as np

# Local packages
from fynance.models.econometric_models import ARMAX_GARCH


def test_armax_external_regressor_applied_via_psi():
    # With no AR/MA/GARCH dynamics, u_t = y_t - c - x_t . psi.
    rng = np.random.default_rng(0)
    T = 12
    y = rng.normal(0, 1, T)
    x = np.full((T, 1), 2.0)          # constant external regressor
    psi = np.array([3.0])             # x-coefficient -> contributes 2*3 = 6
    theta = np.array([0.0])           # MA off
    phi = np.array([0.0])
    alpha = np.array([0.0])
    beta = np.array([0.0])

    u, _ = ARMAX_GARCH(y, x, phi, psi, theta, alpha, beta,
                       c=0.0, omega=1.0, p=0, q=0, Q=0, P=0)

    assert np.allclose(np.asarray(u), y - 6.0, atol=1e-12)


def test_armax_psi_and_theta_are_distinct_roles():
    # psi must drive x, theta must drive past residuals — swapping them changes
    # the result, so the two coefficients are not interchangeable.
    rng = np.random.default_rng(1)
    T = 20
    y = rng.normal(0, 1, T)
    x = rng.normal(0, 1, (T, 1))
    phi = np.array([0.0])
    alpha = np.array([0.0])
    beta = np.array([0.0])

    u1, _ = ARMAX_GARCH(y, x, phi, np.array([0.5]), np.array([0.1]),
                        alpha, beta, 0.0, 1.0, 0, 1, 0, 0)
    u2, _ = ARMAX_GARCH(y, x, phi, np.array([0.1]), np.array([0.5]),
                        alpha, beta, 0.0, 1.0, 0, 1, 0, 0)
    assert not np.allclose(np.asarray(u1), np.asarray(u2))
