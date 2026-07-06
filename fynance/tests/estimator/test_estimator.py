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
    # fail loudly and point to the Numba-backed path instead of returning
    # silently wrong estimates.
    y = np.array([0.1, -0.2, 0.3, 0.0, -0.1])
    with pytest.raises(NotImplementedError, match="get_parameters"):
        estimation(y, x0=np.zeros(2), model="arma")


def test_estimation_message_points_to_fit_volatility():
    # The stub also routes callers to the new MLE volatility driver.
    y = np.array([0.1, -0.2, 0.3, 0.0, -0.1])
    with pytest.raises(NotImplementedError, match="fit_volatility"):
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


def test_loglikelihood_does_not_mutate_input_h():
    # The zero-flooring must happen on a copy: the caller's h is preserved
    # (in particular its zero entries are not overwritten by 1e-8).
    u = np.array([0.0, 1.0, -1.0])
    h = np.array([0.0, 1.0, 2.0])
    h_before = h.copy()
    loglikelihood(u, h)
    assert np.array_equal(h, h_before)
    assert h[0] == 0.0


def test_loglikelihood_returns_negative_log_likelihood():
    # Documented as the negative LL (a cost): it equals -(true Gaussian LL),
    # which for non-degenerate residuals is a positive number.
    rng = np.random.RandomState(0)
    u = rng.randn(50)
    h = np.abs(rng.randn(50)) + 0.5
    true_ll = -0.5 * (
        u.size * np.log(2 * np.pi)
        + np.sum(np.log(np.square(h)))
        + np.sum(np.square(u / h))
    )
    assert np.isclose(loglikelihood(u, h), -true_ll)


def test_target_function_arma_returns_finite():
    import numpy as np

    from fynance.estimator.estimator import target_function

    y = np.random.default_rng(0).normal(0, 1, 40).astype(np.float64)
    # params: c, phi(1), theta(1)
    L = target_function(np.array([0.0, 0.3, 0.2]), y, p=1, q=1, cons=True,
                        model="arma")
    assert np.isfinite(L)


def test_target_function_garch_returns_finite():
    import numpy as np

    from fynance.estimator.estimator import target_function

    y = np.random.default_rng(1).normal(0, 1, 40).astype(np.float64)
    # params: c, phi(1), theta(1), omega, alpha(1), beta(1)
    params = np.array([0.0, 0.3, 0.2, 0.1, 0.1, 0.85])
    L = target_function(params, y, p=1, q=1, Q=1, P=1, cons=True, model="garch")
    assert np.isfinite(L)


def test_target_function_unknown_model_raises():
    import numpy as np
    import pytest

    from fynance.estimator.estimator import target_function

    with pytest.raises(ValueError):
        target_function(np.array([0.0]), np.zeros(5), cons=True, model="nope")
