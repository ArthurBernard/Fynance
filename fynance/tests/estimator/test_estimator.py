#!/usr/bin/env python3
# coding: utf-8

""" Tests for the (pure-Python) estimator helpers. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.estimator.estimator import estimation, loglikelihood


def test_estimation_raises_not_implemented():
    # estimation() is an experimental, non-functional placeholder; it must
    # fail loudly and point to the Cython-backed path instead of returning
    # silently wrong estimates.
    y = np.array([0.1, -0.2, 0.3, 0.0, -0.1])
    with pytest.raises(NotImplementedError, match="get_parameters"):
        estimation(y, x0=np.zeros(2), model="arma")


def test_loglikelihood_matches_formula():
    rng = np.random.RandomState(0)
    u = rng.randn(50)
    h = np.abs(rng.randn(50)) + 0.5
    got = loglikelihood(u.copy(), h.copy())
    T = h.size
    expected = 0.5 * (
        T * np.log(2 * np.pi)
        + np.sum(np.log(np.square(h)))
        + np.sum(np.square(u / h))
    )
    assert np.isclose(got, expected)


def test_loglikelihood_handles_zero_h():
    # zeros in h are floored to 1e-8 (no div-by-zero), result stays finite
    u = np.array([0.0, 1.0, -1.0])
    h = np.array([0.0, 1.0, 2.0])
    assert np.isfinite(loglikelihood(u, h))
