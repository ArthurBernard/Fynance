#!/usr/bin/env python3
# coding: utf-8

""" Test benchmark-relative metrics (beta/alpha/TE/IR/capture). """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.features.roll_functions import roll_beta
from fynance.metrics.benchmark import (
    alpha,
    benchmark_summary,
    beta,
    capture_ratio,
    information_ratio,
    roll_beta_benchmark,
    tracking_error,
)


def _prices_from_returns(r: np.ndarray, p0: float = 100.0) -> np.ndarray:
    """ Build a price curve from a simple-return sample, X[0] = p0. """
    return np.concatenate([[p0], p0 * np.cumprod(1.0 + r)])


# --------------------------------------------------------------------------- #
#                          X identical to the benchmark                       #
# --------------------------------------------------------------------------- #


def test_identical_curves():
    X = np.array([100., 102., 101., 105., 103., 108., 110.])
    B = X.copy()

    assert beta(X, B) == pytest.approx(1.0)
    assert alpha(X, B) == pytest.approx(0.0, abs=1e-10)
    assert tracking_error(X, B) == pytest.approx(0.0, abs=1e-12)
    assert information_ratio(X, B) == 0.0
    assert capture_ratio(X, B, side='up') == pytest.approx(1.0)
    assert capture_ratio(X, B, side='down') == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#                         X-returns = 2 * B-returns                           #
# --------------------------------------------------------------------------- #


def test_double_beta():
    rng = np.random.default_rng(42)
    b_ret = rng.normal(0., 0.01, 999)
    x_ret = 2.0 * b_ret
    B = _prices_from_returns(b_ret)
    X = _prices_from_returns(x_ret)

    assert beta(X, B) == pytest.approx(2.0, rel=1e-9)


# --------------------------------------------------------------------------- #
#                    X = B-returns + constant offset c                        #
# --------------------------------------------------------------------------- #


def test_constant_offset_alpha_and_beta():
    rng = np.random.default_rng(7)
    b_ret = rng.normal(0., 0.01, 999)
    c = 0.0005
    x_ret = b_ret + c
    B = _prices_from_returns(b_ret)
    X = _prices_from_returns(x_ret)

    assert beta(X, B) == pytest.approx(1.0, abs=1e-4)
    assert alpha(X, B, period=252) == pytest.approx(c * 252, rel=1e-2)
    assert information_ratio(X, B, period=252) > 0.


# --------------------------------------------------------------------------- #
#                    Hand-built 6-bar up/down capture case                    #
# --------------------------------------------------------------------------- #


def test_hand_built_capture_ratio():
    # 5 "real" bars (indices 1..5); index 0 is the synthetic R_1 = 0 bar shared
    # by X and B, so it lands in neither the up- nor the down-bar mask.
    b_ret_real = np.array([0.10, -0.05, 0.20, -0.10, 0.05])
    x_ret_real = np.array([0.05, -0.10, 0.30, -0.05, 0.10])
    B = _prices_from_returns(b_ret_real)
    X = _prices_from_returns(x_ret_real)

    period = 252

    # Independent reference computation (mirrors the docstring formula, but
    # written from scratch rather than reusing the module's private helpers).
    up = b_ret_real > 0.
    down = b_ret_real < 0.
    n_up = int(up.sum())
    n_down = int(down.sum())
    expected_up = (
        np.prod(1. + x_ret_real[up]) ** (period / n_up) - 1.
    ) / (
        np.prod(1. + b_ret_real[up]) ** (period / n_up) - 1.
    )
    expected_down = (
        np.prod(1. + x_ret_real[down]) ** (period / n_down) - 1.
    ) / (
        np.prod(1. + b_ret_real[down]) ** (period / n_down) - 1.
    )

    assert capture_ratio(X, B, side='up', period=period) == pytest.approx(expected_up)
    assert capture_ratio(X, B, side='down', period=period) == pytest.approx(expected_down)


# --------------------------------------------------------------------------- #
#                                roll_beta parity                             #
# --------------------------------------------------------------------------- #


def test_roll_beta_benchmark_matches_roll_beta_on_derived_returns():
    rng = np.random.default_rng(3)
    b_ret = rng.normal(0., 0.01, 500)
    x_ret = 0.7 * b_ret + rng.normal(0., 0.001, 500)
    B = _prices_from_returns(b_ret)
    X = _prices_from_returns(x_ret)

    w = 63
    got = roll_beta_benchmark(X, B, w=w)

    # Reference: derive the same simple returns by hand and call roll_beta.
    x_full = np.concatenate([[0.], X[1:] / X[:-1] - 1.])
    b_full = np.concatenate([[0.], B[1:] / B[:-1] - 1.])
    expected = roll_beta(x_full, b_full, w)

    np.testing.assert_allclose(got, expected, equal_nan=True)
    assert np.isnan(got[:w - 1]).all()


# --------------------------------------------------------------------------- #
#                                   Errors                                    #
# --------------------------------------------------------------------------- #


def test_length_mismatch_raises_value_error():
    X = np.array([100., 101., 102.])
    B = np.array([100., 101.])

    with pytest.raises(ValueError):
        beta(X, B)

    with pytest.raises(ValueError):
        alpha(X, B)

    with pytest.raises(ValueError):
        tracking_error(X, B)

    with pytest.raises(ValueError):
        information_ratio(X, B)

    with pytest.raises(ValueError):
        capture_ratio(X, B)

    with pytest.raises(ValueError):
        benchmark_summary(X, B)

    with pytest.raises(ValueError):
        roll_beta_benchmark(X, B)


def test_invalid_side_raises_value_error():
    X = np.array([100., 101., 102.])
    B = np.array([100., 102., 101.])

    with pytest.raises(ValueError):
        capture_ratio(X, B, side='sideways')


# --------------------------------------------------------------------------- #
#                              benchmark_summary                              #
# --------------------------------------------------------------------------- #


def test_benchmark_summary_keys_complete():
    rng = np.random.default_rng(11)
    b_ret = rng.normal(0., 0.01, 300)
    x_ret = 0.5 * b_ret + rng.normal(0., 0.002, 300)
    B = _prices_from_returns(b_ret)
    X = _prices_from_returns(x_ret)

    s = benchmark_summary(X, B)

    assert set(s) == {
        'beta', 'alpha', 'tracking_error', 'information_ratio',
        'up_capture', 'down_capture',
    }
    assert s['beta'] == pytest.approx(beta(X, B))
    assert s['alpha'] == pytest.approx(alpha(X, B))
    assert s['tracking_error'] == pytest.approx(tracking_error(X, B))
    assert s['information_ratio'] == pytest.approx(information_ratio(X, B))
    assert s['up_capture'] == pytest.approx(capture_ratio(X, B, side='up'))
    assert s['down_capture'] == pytest.approx(capture_ratio(X, B, side='down'))
