#!/usr/bin/env python3
# coding: utf-8

""" Tests for the NaN-aware cross-sectional transforms (features.cross_section). """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.features.cross_section import (
    cs_demean,
    cs_neutralize,
    cs_rank,
    cs_winsorize,
    cs_zscore,
)


def _synthetic_panel(T=200, N=8, nan_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))
    n_nan = int(round(nan_frac * T * N))
    flat_idx = rng.choice(T * N, size=n_nan, replace=False)
    X.reshape(-1)[flat_idx] = np.nan

    return X


# -- Hand values ------------------------------------------------------------


def test_cs_rank_hand_values():
    X = np.array([[3., 1., 2.]])
    out = cs_rank(X, pct=True)
    assert np.allclose(out, [[1.0, 0.0, 0.5]])


def test_cs_rank_raw_hand_values():
    X = np.array([[3., 1., 2.]])
    out = cs_rank(X, pct=False)
    assert np.allclose(out, [[3., 1., 2.]])


def test_cs_zscore_hand_values():
    X = np.array([[1., 2., 3.]])
    out = cs_zscore(X)
    mean = 2.
    std = np.std([1., 2., 3.])
    expected = (np.array([1., 2., 3.]) - mean) / std
    assert np.allclose(out, [expected])


def test_cs_winsorize_hand_values():
    X = np.array([[1., 2., 3., 4., 100.]])
    out = cs_winsorize(X, alpha=0.2)
    lo, hi = np.quantile(X[0], [0.2, 0.8])
    expected = np.clip(X[0], lo, hi)
    assert np.allclose(out, [expected])


def test_cs_demean_mean_is_zero():
    X = np.array([[1., 2., 3.]])
    out = cs_demean(X)
    assert np.allclose(out, [[-1., 0., 1.]])
    assert np.mean(out) == pytest.approx(0.)


# -- NaN-awareness ------------------------------------------------------------


def test_cs_rank_nan_preserved():
    X = np.array([[1., np.nan, 3.]])
    out = cs_rank(X)
    assert np.isnan(out[0, 1])
    assert not np.isnan(out[0, 0])
    assert not np.isnan(out[0, 2])
    # only 2 valid entries -> ranks 1 and 2 rescaled to 0 and 1
    assert out[0, 0] == pytest.approx(0.)
    assert out[0, 2] == pytest.approx(1.)


def test_cs_zscore_nan_preserved_stats_over_valid_only():
    X = np.array([[1., np.nan, 3.]])
    out = cs_zscore(X)
    assert np.isnan(out[0, 1])
    valid_mean = np.mean([1., 3.])
    valid_std = np.std([1., 3.])
    expected = (np.array([1., 3.]) - valid_mean) / valid_std
    assert out[0, 0] == pytest.approx(expected[0])
    assert out[0, 2] == pytest.approx(expected[1])


def test_cs_demean_nan_preserved():
    X = np.array([[1., np.nan, 3.]])
    out = cs_demean(X)
    assert np.isnan(out[0, 1])
    assert np.allclose([out[0, 0], out[0, 2]], [-1., 1.])


def test_cs_winsorize_nan_preserved():
    X = np.array([[1., np.nan, 3., 100.]])
    out = cs_winsorize(X, alpha=0.25)
    assert np.isnan(out[0, 1])
    valid = np.array([1., 3., 100.])
    lo, hi = np.quantile(valid, [0.25, 0.75])
    expected = np.clip(valid, lo, hi)
    assert out[0, 0] == pytest.approx(expected[0])
    assert out[0, 2] == pytest.approx(expected[1])
    assert out[0, 3] == pytest.approx(expected[2])


def test_cs_neutralize_nan_preserved():
    X = np.array([[1., np.nan, 3., 4.]])
    f = np.array([[1., 2., 3., 4.]])
    out = cs_neutralize(X, f)
    assert np.isnan(out[0, 1])
    assert not np.isnan(out[0, 0])
    assert not np.isnan(out[0, 2])
    assert not np.isnan(out[0, 3])


# -- Properties on a seeded synthetic panel ----------------------------------


def test_cs_rank_property_bounds():
    X = _synthetic_panel()
    out = cs_rank(X, pct=True)
    mask = ~np.isnan(X)
    assert np.isnan(out[~mask]).all()
    valid_out = out[mask]
    assert valid_out.min() >= 0.
    assert valid_out.max() <= 1.


def test_cs_zscore_property_mean_std():
    X = _synthetic_panel()
    out = cs_zscore(X, ddof=0)
    for t in range(X.shape[0]):
        valid = out[t][~np.isnan(X[t])]
        if valid.size < 2:
            continue
        assert valid.mean() == pytest.approx(0., abs=1e-8)
        assert valid.std(ddof=0) == pytest.approx(1., abs=1e-6)


def test_cs_demean_property_mean_zero():
    X = _synthetic_panel()
    out = cs_demean(X)
    for t in range(X.shape[0]):
        valid = out[t][~np.isnan(X[t])]
        if valid.size == 0:
            continue
        assert valid.mean() == pytest.approx(0., abs=1e-8)


def test_cs_winsorize_property_bounds_match_quantiles():
    X = _synthetic_panel()
    out = cs_winsorize(X, alpha=0.05)
    for t in range(X.shape[0]):
        valid = X[t][~np.isnan(X[t])]
        if valid.size == 0:
            continue
        lo, hi = np.quantile(valid, [0.05, 0.95])
        out_valid = out[t][~np.isnan(X[t])]
        assert out_valid.min() >= lo - 1e-10
        assert out_valid.max() <= hi + 1e-10


def test_cs_neutralize_property_residuals_orthogonal():
    rng = np.random.default_rng(1)
    X = _synthetic_panel()
    exposures = rng.standard_normal(X.shape)
    out = cs_neutralize(X, exposures)
    for t in range(X.shape[0]):
        mask = ~np.isnan(X[t]) & ~np.isnan(exposures[t])
        n_valid = mask.sum()
        if n_valid < 2:
            continue
        resid = out[t][mask]
        if np.isnan(resid).any():
            continue
        expo = exposures[t][mask]
        # Orthogonality: correlation between residual and exposure ~ 0.
        if np.std(resid) == 0. or np.std(expo) == 0.:
            continue
        corr = np.corrcoef(resid, expo)[0, 1]
        assert abs(corr) < 1e-8


# -- Row independence ---------------------------------------------------------


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (cs_rank, {}),
        (cs_zscore, {}),
        (cs_demean, {}),
        (cs_winsorize, {}),
    ],
)
def test_row_independence(func, kwargs):
    X = _synthetic_panel()
    out_before = func(X, **kwargs)
    X2 = X.copy()
    t0 = 5
    X2[t0] = X2[t0] * 2. + 1.
    out_after = func(X2, **kwargs)

    other_rows = np.array([t for t in range(X.shape[0]) if t != t0])
    before_others = out_before[other_rows]
    after_others = out_after[other_rows]
    both_nan = np.isnan(before_others) & np.isnan(after_others)
    assert np.allclose(
        np.where(both_nan, 0., before_others),
        np.where(both_nan, 0., after_others),
    )


def test_row_independence_neutralize():
    rng = np.random.default_rng(2)
    X = _synthetic_panel()
    exposures = rng.standard_normal(X.shape)
    out_before = cs_neutralize(X, exposures)

    X2 = X.copy()
    t0 = 5
    X2[t0] = X2[t0] * 2. + 1.
    out_after = cs_neutralize(X2, exposures)

    other_rows = np.array([t for t in range(X.shape[0]) if t != t0])
    before_others = out_before[other_rows]
    after_others = out_after[other_rows]
    both_nan = np.isnan(before_others) & np.isnan(after_others)
    assert np.allclose(
        np.where(both_nan, 0., before_others),
        np.where(both_nan, 0., after_others),
    )


# -- Ties ---------------------------------------------------------------------


def test_cs_rank_ties_average():
    X = np.array([[1., 1., 2.]])
    out = cs_rank(X, pct=False)
    assert np.allclose(out, [[1.5, 1.5, 3.]])


# -- Errors ---------------------------------------------------------------------


@pytest.mark.parametrize("func", [cs_rank, cs_zscore, cs_demean, cs_winsorize])
def test_1d_input_raises(func):
    with pytest.raises(ValueError):
        func(np.array([1., 2., 3.]))


def test_neutralize_1d_input_raises():
    with pytest.raises(ValueError):
        cs_neutralize(np.array([1., 2., 3.]), np.array([1., 2., 3.]))


def test_neutralize_wrong_shape_exposures_2d_raises():
    X = np.zeros((4, 3))
    exposures = np.zeros((4, 2))  # wrong N
    with pytest.raises(ValueError):
        cs_neutralize(X, exposures)


def test_neutralize_wrong_shape_exposures_3d_raises():
    X = np.zeros((4, 3))
    exposures = np.zeros((4, 2, 2))  # wrong (T, N) leading dims
    with pytest.raises(ValueError):
        cs_neutralize(X, exposures)


def test_neutralize_wrong_ndim_exposures_raises():
    X = np.zeros((4, 3))
    exposures = np.zeros((4, 3, 2, 1))  # 4-D, unsupported
    with pytest.raises(ValueError):
        cs_neutralize(X, exposures)


def test_demean_wrong_weights_shape_raises():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError):
        cs_demean(X, weights=np.zeros(2))


def test_winsorize_invalid_alpha_raises():
    X = np.zeros((2, 3))
    with pytest.raises(ValueError):
        cs_winsorize(X, alpha=0.5)
    with pytest.raises(ValueError):
        cs_winsorize(X, alpha=-0.1)


def test_neutralize_too_few_valid_assets_is_nan():
    X = np.array([[1., np.nan, np.nan]])
    f = np.array([[1., 2., 3.]])
    out = cs_neutralize(X, f)
    assert np.isnan(out).all()
