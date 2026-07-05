#!/usr/bin/env python3
# coding: utf-8

""" Test pairwise rolling statistics (roll_cov/roll_corr/roll_beta/cross_corr). """

# Third-party packages
import numpy as np
import pytest

# Local packages
import fynance as fy
from fynance.features.roll_functions import cross_corr, roll_beta, roll_corr, roll_cov

# --------------------------------------------------------------------------- #
#                    Slow, independent pure-Python reference                  #
# --------------------------------------------------------------------------- #


def _ref_roll_cov(x, y, w):
    T = x.shape[0]
    out = np.full(T, np.nan)
    for t in range(w - 1, T):
        xw = x[t - w + 1: t + 1]
        yw = y[t - w + 1: t + 1]
        mx = sum(xw) / w
        my = sum(yw) / w
        s = sum((xi - mx) * (yi - my) for xi, yi in zip(xw, yw))
        out[t] = s / w
    return out


def _ref_roll_corr(x, y, w):
    T = x.shape[0]
    out = np.full(T, np.nan)
    for t in range(w - 1, T):
        xw = x[t - w + 1: t + 1]
        yw = y[t - w + 1: t + 1]
        mx = sum(xw) / w
        my = sum(yw) / w
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(xw, yw))
        sxx = sum((xi - mx) ** 2 for xi in xw)
        syy = sum((yi - my) ** 2 for yi in yw)
        denom = (sxx * syy) ** 0.5
        out[t] = np.nan if denom == 0. else sxy / denom
    return out


def _ref_roll_beta(x, y, w):
    T = x.shape[0]
    out = np.full(T, np.nan)
    for t in range(w - 1, T):
        xw = x[t - w + 1: t + 1]
        yw = y[t - w + 1: t + 1]
        mx = sum(xw) / w
        my = sum(yw) / w
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(xw, yw))
        syy = sum((yi - my) ** 2 for yi in yw)
        out[t] = np.nan if syy == 0. else sxy / syy
    return out


def _ref_cross_corr(x, y, max_lag):
    T = x.shape[0]
    out = np.full(2 * max_lag + 1, np.nan)
    for idx, lag in enumerate(range(-max_lag, max_lag + 1)):
        # corr(x[t], y[t - lag]) over all valid t.
        if lag >= 0:
            xw = x[lag:]
            yw = y[: T - lag]
        else:
            k = -lag
            xw = x[: T - k]
            yw = y[k:]
        n = len(xw)
        mx = sum(xw) / n
        my = sum(yw) / n
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(xw, yw))
        sxx = sum((xi - mx) ** 2 for xi in xw)
        syy = sum((yi - my) ** 2 for yi in yw)
        denom = (sxx * syy) ** 0.5
        out[idx] = np.nan if denom == 0. else sxy / denom
    return out


@pytest.fixture()
def series_pair():
    rng = np.random.default_rng(123)
    x = rng.normal(size=500)
    y = 0.4 * x + rng.normal(scale=1.3, size=500)
    return x, y


# --------------------------------------------------------------------------- #
#                              Reference parity                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('w', [2, 5, 20, 63])
def test_roll_cov_matches_reference(series_pair, w):
    x, y = series_pair
    out = roll_cov(x, y, w=w)
    expected = _ref_roll_cov(x, y, w)
    np.testing.assert_allclose(out, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize('w', [2, 5, 20, 63])
def test_roll_corr_matches_reference(series_pair, w):
    x, y = series_pair
    out = roll_corr(x, y, w=w)
    expected = _ref_roll_corr(x, y, w)
    np.testing.assert_allclose(out, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize('w', [2, 5, 20, 63])
def test_roll_beta_matches_reference(series_pair, w):
    x, y = series_pair
    out = roll_beta(x, y, w=w)
    expected = _ref_roll_beta(x, y, w)
    np.testing.assert_allclose(out, expected, rtol=1e-12, equal_nan=True)


def test_cross_corr_matches_reference(series_pair):
    x, y = series_pair
    max_lag = 15
    out = cross_corr(x, y, max_lag=max_lag)
    expected = _ref_cross_corr(x, y, max_lag)
    np.testing.assert_allclose(out, expected, rtol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
#                                Known values                                 #
# --------------------------------------------------------------------------- #


def test_roll_corr_of_y_equal_2x_is_one():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    y = 2. * x
    out = roll_corr(x, y, w=10)
    assert np.all(np.isnan(out[:9]))
    assert out[9:] == pytest.approx(1.0)


def test_roll_beta_of_y_equal_2x_is_half():
    # roll_beta(x, y) = cov(x, y) / var(y); for y = 2x, cov(x, y) = 2 var(x)
    # and var(y) = 4 var(x), so beta = 2 var(x) / (4 var(x)) = 0.5.
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    y = 2. * x
    out = roll_beta(x, y, w=10)
    assert np.all(np.isnan(out[:9]))
    assert out[9:] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#                                  Causality                                  #
# --------------------------------------------------------------------------- #


def test_roll_cov_corr_beta_are_causal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    w = 20
    t0 = 100

    base_cov, base_corr, base_beta = roll_cov(x, y, w), roll_corr(x, y, w), roll_beta(x, y, w)

    x2 = x.copy()
    y2 = y.copy()
    x2[t0:] += rng.normal(size=200 - t0)
    y2[t0:] += rng.normal(size=200 - t0)

    new_cov, new_corr, new_beta = roll_cov(x2, y2, w), roll_corr(x2, y2, w), roll_beta(x2, y2, w)

    # Everything strictly before t0 is untouched by a perturbation at/after t0.
    np.testing.assert_array_equal(base_cov[:t0], new_cov[:t0])
    np.testing.assert_array_equal(base_corr[:t0], new_corr[:t0])
    np.testing.assert_array_equal(base_beta[:t0], new_beta[:t0])
    # Sanity: the perturbation does actually change later outputs.
    assert not np.allclose(base_cov[t0:], new_cov[t0:], equal_nan=True)


# --------------------------------------------------------------------------- #
#                          NaN head / zero variance                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('w', [2, 7, 30])
def test_nan_head_length_is_w_minus_1(series_pair, w):
    x, y = series_pair
    for out in (roll_cov(x, y, w=w), roll_corr(x, y, w=w), roll_beta(x, y, w=w)):
        assert np.all(np.isnan(out[: w - 1]))
        assert np.all(np.isfinite(out[w - 1:]))


def test_zero_variance_stretch_gives_nan_no_warnings(recwarn):
    # roll_corr is symmetric in x/y: zero variance in *either* side gives NaN.
    x_flat = np.concatenate([np.full(20, 3.0), np.random.default_rng(4).normal(size=20)])
    y = np.random.default_rng(5).normal(size=40)
    w = 10
    corr = roll_corr(x_flat, y, w=w)
    assert np.all(np.isnan(corr[w - 1:20]))

    # roll_beta = cov(x, y) / var(y): only zero variance in the *denominator*
    # (y) forces NaN — zero variance in x alone gives beta = 0 (cov = 0), not
    # NaN, since var(y) != 0 there.
    beta_x_flat = roll_beta(x_flat, y, w=w)
    assert np.all(np.isfinite(beta_x_flat[w - 1:20]))
    assert beta_x_flat[w - 1:20] == pytest.approx(0.0)

    y_flat = np.concatenate([np.full(20, -1.0), np.random.default_rng(6).normal(size=20)])
    x = np.random.default_rng(7).normal(size=40)
    beta_y_flat = roll_beta(x, y_flat, w=w)
    assert np.all(np.isnan(beta_y_flat[w - 1:20]))

    assert len(recwarn) == 0


# --------------------------------------------------------------------------- #
#                                  cross_corr                                 #
# --------------------------------------------------------------------------- #


def test_cross_corr_argmax_matches_documented_lag_convention():
    # y[t] = x[t - 3] + tiny noise: x leads y by 3 bars. Per the documented
    # convention (entry `lag` correlates x[t] with y[t - lag]), the profile
    # must peak at lag = -3, since y[t - (-3)] = y[t + 3] = x[t].
    rng = np.random.default_rng(6)
    x = rng.normal(size=1000)
    noise = rng.normal(scale=1e-6, size=1000)
    y = np.empty(1000)
    y[:3] = rng.normal(size=3)
    y[3:] = x[:-3] + noise[3:]

    max_lag = 10
    profile = cross_corr(x, y, max_lag=max_lag)
    lags = np.arange(-max_lag, max_lag + 1)
    assert lags[np.argmax(profile)] == -3


def test_cross_corr_shape_and_symmetry_indexing():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    max_lag = 8
    profile = cross_corr(x, y, max_lag=max_lag)
    assert profile.shape == (2 * max_lag + 1,)
    # lag=0 is corr(x, y) computed over the full overlap (all T points).
    expected_lag0 = np.corrcoef(x, y)[0, 1]
    assert profile[max_lag] == pytest.approx(expected_lag0)


# --------------------------------------------------------------------------- #
#                                   Errors                                    #
# --------------------------------------------------------------------------- #


def test_length_mismatch_raises():
    x = np.array([1., 2., 3.])
    y = np.array([1., 2.])
    with pytest.raises(ValueError):
        roll_cov(x, y, w=2)
    with pytest.raises(ValueError):
        roll_corr(x, y, w=2)
    with pytest.raises(ValueError):
        roll_beta(x, y, w=2)
    with pytest.raises(ValueError):
        cross_corr(x, y, max_lag=1)


def test_window_below_two_raises():
    x = np.array([1., 2., 3., 4.])
    y = np.array([4., 3., 2., 1.])
    with pytest.raises(ValueError):
        roll_cov(x, y, w=1)
    with pytest.raises(ValueError):
        roll_corr(x, y, w=1)
    with pytest.raises(ValueError):
        roll_beta(x, y, w=1)


def test_max_lag_out_of_bounds_raises():
    x = np.arange(5.)
    y = np.arange(5.)
    with pytest.raises(ValueError):
        cross_corr(x, y, max_lag=5)
    with pytest.raises(ValueError):
        cross_corr(x, y, max_lag=6)


def test_nan_input_raises():
    x = np.array([1., np.nan, 3.])
    y = np.array([1., 2., 3.])
    with pytest.raises(ValueError):
        roll_cov(x, y, w=2)
    with pytest.raises(ValueError):
        cross_corr(x, y, max_lag=1)


def test_exposed_on_top_level_package():
    assert fy.roll_cov is roll_cov
    assert fy.roll_corr is roll_corr
    assert fy.roll_beta is roll_beta
    assert fy.cross_corr is cross_corr
