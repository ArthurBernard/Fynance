#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.7 feature-engineering tools. """

import numpy as np
import pytest

from fynance.features.engineering import (
    IncrementalMoments,
    _fracdiff_weights,
    fracdiff,
    granger_causality,
    multi_resolution,
)
from fynance.features.momentums import sma


def test_multi_resolution_shape_and_columns():
    X = np.arange(1.0, 11.0)
    out = multi_resolution(sma, X, [2, 3])
    assert out.shape == (10, 2)
    assert np.allclose(out[:, 0], np.asarray(sma(X, 2)).reshape(-1))
    assert np.allclose(out[:, 1], np.asarray(sma(X, 3)).reshape(-1))


def test_granger_detects_causality():
    rng = np.random.RandomState(0)
    x = rng.standard_normal(300)
    y = np.r_[0.0, 0.8 * x[:-1]] + 0.1 * rng.standard_normal(300)
    _, p = granger_causality(x, y, lag=1)
    assert p < 0.01


def test_granger_independent_not_significant():
    rng = np.random.RandomState(1)
    _, p = granger_causality(rng.standard_normal(300), rng.standard_normal(300), lag=1)
    assert p > 0.05


def test_granger_too_short_raises():
    with pytest.raises(ValueError):
        granger_causality(np.arange(3.0), np.arange(3.0), lag=1)


def test_granger_detects_causality_lag2():
    # y_t driven by x_{t-2}: lag=2 must pick up the two-step-ahead causality.
    rng = np.random.RandomState(5)
    x = rng.standard_normal(400)
    y = np.r_[0.0, 0.0, 0.7 * x[:-2]] + 0.1 * rng.standard_normal(400)
    f, p = granger_causality(x, y, lag=2)
    assert f > 0.0
    assert p < 0.01


def test_granger_independent_not_significant_lag2():
    rng = np.random.RandomState(0)
    _, p = granger_causality(
        rng.standard_normal(400), rng.standard_normal(400), lag=2
    )
    assert p > 0.05


def test_incremental_moments_matches_numpy():
    rng = np.random.RandomState(2)
    data = rng.standard_normal(100)
    im = IncrementalMoments()
    for v in data:
        im.update(v)
    assert im.n == 100
    assert np.isclose(im.mean, data.mean())
    assert np.isclose(im.var, data.var())
    assert np.isclose(im.std, data.std())


def test_incremental_update_returns_self():
    im = IncrementalMoments()
    assert im.update(1.0) is im


def test_incremental_single_observation_zero_variance():
    # After a single observation the variance/std are well-defined and zero (no
    # division-by-zero, no nan), and the mean equals that observation.
    im = IncrementalMoments()
    im.update(3.5)
    assert im.n == 1
    assert im.mean == 3.5
    assert im.var == 0.0
    assert im.std == 0.0


def test_incremental_empty_is_zero():
    # Before any observation, the moments default to zero rather than raising.
    im = IncrementalMoments()
    assert im.n == 0
    assert im.mean == 0.0
    assert im.var == 0.0
    assert im.std == 0.0


def test_fracdiff_weights_known_values_d_half():
    w = _fracdiff_weights(0.5, tol=1e-5)
    expected_head = np.array([1.0, -0.5, -0.125, -0.0625])
    np.testing.assert_allclose(w[:4], expected_head, rtol=1e-12)


def test_fracdiff_weights_always_keep_w0():
    # d=0 makes w_1 vanish exactly, but w_0 = 1 must always be kept.
    w = _fracdiff_weights(0.0, tol=1e-5)
    assert w.shape == (1,)
    assert w[0] == 1.0


def test_fracdiff_d0_is_identity_post_warmup():
    X = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    y = fracdiff(X, d=0.0)
    assert np.array_equal(y, X)


def test_fracdiff_d1_equals_first_difference_post_warmup():
    X = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    y = fracdiff(X, d=1.0)
    assert np.isnan(y[0])
    np.testing.assert_array_equal(y[1:], np.diff(X))


def _slow_fracdiff(X, d, tol):
    """ Reference double-loop implementation, independent of the kernel. """
    w = _fracdiff_weights(d, tol)
    K = len(w)
    T, N = X.shape
    out = np.full((T, N), np.nan)
    for n in range(N):
        for t in range(K - 1, T):
            s = 0.0
            for k in range(K):
                s += w[k] * X[t - k, n]
            out[t, n] = s

    return out


def test_fracdiff_matches_slow_python_reference():
    # tol=1e-3 (rather than the 1e-5 default) keeps K well below T=300, so the
    # convolution is actually exercised rather than degenerating to all-NaN.
    rng = np.random.RandomState(7)
    X = 100 * np.exp(np.cumsum(rng.standard_normal((300, 3)) * 0.01, axis=0))
    d, tol = 0.4, 1e-3
    fast = fracdiff(X, d=d, tol=tol)
    slow = _slow_fracdiff(X, d, tol)
    np.testing.assert_allclose(fast, slow, rtol=1e-12, equal_nan=True)


def test_fracdiff_is_causal():
    rng = np.random.RandomState(3)
    X = 100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
    t0 = 100
    X_perturbed = X.copy()
    X_perturbed[t0:] += rng.standard_normal(X.shape[0] - t0)

    y = fracdiff(X, d=0.4, tol=1e-3)
    y_perturbed = fracdiff(X_perturbed, d=0.4, tol=1e-3)

    np.testing.assert_array_equal(y[:t0], y_perturbed[:t0])
    assert not np.allclose(y[t0:], y_perturbed[t0:], equal_nan=True)


def test_fracdiff_nan_head_length():
    rng = np.random.RandomState(4)
    tol = 1e-3
    K = _fracdiff_weights(0.4, tol).shape[0]
    X = rng.standard_normal(K + 50) + 100
    y = fracdiff(X, d=0.4, tol=tol)
    assert np.all(np.isnan(y[:K - 1]))
    assert np.all(np.isfinite(y[K - 1:]))


def test_fracdiff_short_series_all_nan():
    tol = 1e-3
    K = _fracdiff_weights(0.4, tol).shape[0]
    X = np.arange(1.0, float(K))  # length K - 1 < K
    y = fracdiff(X, d=0.4, tol=tol)
    assert y.shape == (K - 1,)
    assert np.all(np.isnan(y))


def test_fracdiff_2d_matches_stacked_1d():
    rng = np.random.RandomState(9)
    X = rng.standard_normal((100, 3)).cumsum(axis=0) + 100
    y2d = fracdiff(X, d=0.4, tol=1e-3)
    stacked = np.column_stack(
        [fracdiff(X[:, j], d=0.4, tol=1e-3) for j in range(3)]
    )
    np.testing.assert_array_equal(y2d, stacked)


@pytest.mark.parametrize("d", [-0.1, 2.1])
def test_fracdiff_invalid_d_raises(d):
    with pytest.raises(ValueError):
        fracdiff(np.arange(10.0), d=d)


def test_fracdiff_nan_input_raises():
    X = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError):
        fracdiff(X)


def test_fracdiff_inf_input_raises():
    X = np.array([1.0, 2.0, np.inf, 4.0])
    with pytest.raises(ValueError):
        fracdiff(X)
