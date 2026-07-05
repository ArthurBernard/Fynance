#!/usr/bin/env python3
# coding: utf-8

""" Tests for the Alphalens-style factor-evaluation suite. """

# Third-party packages
import numpy as np
import pytest

# Local packages
import fynance as fy
from fynance.metrics import (
    QuantileResult,
    factor_rank_autocorr,
    ic_decay,
    ic_summary,
    information_coefficient,
    quantile_returns,
    roll_information_coefficient,
)

# ---------------------------------------------------------------------------
# quantile_returns
# ---------------------------------------------------------------------------

def test_quantile_returns_hand_built_panel():
    # Two-bar, six-asset panel, three equal-count buckets. Bar 0 has a monotone
    # factor so the buckets are {a0,a1}, {a2,a3}, {a4,a5}; bar 1 has only two
    # valid assets (< Q=3) so it must be a NaN row.
    factor = np.array([[1., 2., 3., 4., 5., 6.],
                       [5., 6., np.nan, np.nan, np.nan, np.nan]])
    fwd = np.array([[10., 20., 30., 40., 50., 60.],
                    [1., 1., 1., 1., 1., 1.]])
    res = quantile_returns(factor, fwd, n_quantiles=3)

    assert isinstance(res, QuantileResult)
    assert res.n_quantiles == 3
    # Bar 0 bucket means and counts.
    assert np.allclose(res.quantile_returns[0], [15., 35., 55.])
    assert res.spread[0] == pytest.approx(40.0)
    assert np.array_equal(res.counts[0], [2, 2, 2])
    # Bar 1: fewer than Q valid assets -> NaN row, zero counts, NaN spread.
    assert np.all(np.isnan(res.quantile_returns[1]))
    assert np.array_equal(res.counts[1], [0, 0, 0])
    assert np.isnan(res.spread[1])


def test_quantile_returns_ties_broken_by_rank_order():
    # All-equal factor: a stable sort keeps original order, filling buckets
    # left-to-right, so counts stay balanced even under total ties.
    factor = np.zeros((1, 6))
    fwd = np.arange(6.).reshape(1, 6)
    res = quantile_returns(factor, fwd, n_quantiles=3)
    assert np.array_equal(res.counts[0], [2, 2, 2])


def test_quantile_returns_rejects_bad_n_quantiles():
    panel = np.zeros((2, 5))
    with pytest.raises(ValueError):
        quantile_returns(panel, panel, n_quantiles=1)


def test_quantile_returns_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        quantile_returns(np.zeros((2, 5)), np.zeros((2, 4)))


# ---------------------------------------------------------------------------
# Perfect foresight: factor == forward return
# ---------------------------------------------------------------------------

def _gbm_panel(T=300, N=20, seed=0):
    """ Seeded GBM price panel and the aligned one-bar forward return. """
    rng = np.random.default_rng(seed)
    prices = 100. * np.cumprod(1. + rng.normal(0., 0.01, (T, N)), axis=0)
    fwd = np.full((T, N), np.nan)
    fwd[:-1] = prices[1:] / prices[:-1] - 1.

    return prices, fwd


def test_perfect_foresight_top_beats_bottom_every_bar():
    _, fwd = _gbm_panel()
    factor = fwd.copy()
    res = quantile_returns(factor, fwd, n_quantiles=5)
    valid = np.isfinite(res.spread)
    # Sorting by the factor sorts by the realized return, so the top bucket
    # out-earns the bottom bucket on every valid bar.
    assert np.all(res.quantile_returns[valid, -1] > res.quantile_returns[valid, 0])
    assert np.all(res.spread[valid] > 0.0)


def test_perfect_foresight_ic_summary_mean_ic_is_one():
    _, fwd = _gbm_panel()
    factor = fwd.copy()
    s = ic_summary(factor, fwd, method='spearman')
    assert s['mean_ic'] == pytest.approx(1.0)
    assert s['hit_rate'] == pytest.approx(1.0)
    assert s['n_bars'] == 299  # last bar is all-NaN and dropped
    assert isinstance(s['n_bars'], int)


def test_perfect_foresight_ic_decay_is_one_at_horizon_one():
    prices, fwd = _gbm_panel()
    factor = fwd.copy()
    decay = ic_decay(factor, prices, horizons=(1, 5, 10, 21))
    assert decay.shape == (4,)
    assert decay[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Noise factor: no signal
# ---------------------------------------------------------------------------

def test_noise_factor_has_no_signal():
    rng = np.random.default_rng(7)
    factor = rng.standard_normal((500, 30))
    fwd = rng.standard_normal((500, 30))  # independent of factor
    s = ic_summary(factor, fwd, method='spearman')
    assert abs(s['mean_ic']) < 0.1
    assert 0.35 <= s['hit_rate'] <= 0.65


def test_ic_summary_empty_panel_returns_nan():
    factor = np.full((5, 4), np.nan)
    s = ic_summary(factor, factor)
    assert s['n_bars'] == 0
    assert np.isnan(s['mean_ic'])


# ---------------------------------------------------------------------------
# roll_information_coefficient
# ---------------------------------------------------------------------------

def test_roll_ic_1d_matches_slow_loop():
    rng = np.random.default_rng(3)
    pred = rng.standard_normal(200)
    real = pred + 0.5 * rng.standard_normal(200)
    w = 30
    fast = roll_information_coefficient(pred, real, w=w, method='spearman')

    slow = np.full(200, np.nan)
    for t in range(w - 1, 200):
        slow[t] = information_coefficient(
            pred[t - w + 1:t + 1], real[t - w + 1:t + 1], method='spearman')

    assert np.allclose(fast[w - 1:], slow[w - 1:], rtol=1e-10, atol=0.0)
    assert np.all(np.isnan(fast[:w - 1]))


def test_roll_ic_2d_has_nan_head():
    rng = np.random.default_rng(4)
    pred = rng.standard_normal((100, 10))
    real = rng.standard_normal((100, 10))
    w = 20
    ic = roll_information_coefficient(pred, real, w=w)
    assert ic.shape == (100,)
    assert np.all(np.isnan(ic[:w - 1]))
    assert np.all(np.isfinite(ic[w - 1:]))


def test_roll_ic_is_causal():
    # Perturbing a future bar must not change any earlier trailing-window IC.
    rng = np.random.default_rng(5)
    pred = rng.standard_normal(300)
    real = pred + 0.3 * rng.standard_normal(300)
    w = 40
    before = roll_information_coefficient(pred, real, w=w)

    t0 = 200
    pred2 = pred.copy()
    pred2[t0:] += 5.0  # corrupt the future only
    after = roll_information_coefficient(pred2, real, w=w)

    assert np.allclose(before[:t0], after[:t0], equal_nan=True)


def test_roll_ic_rejects_bad_window():
    with pytest.raises(ValueError):
        roll_information_coefficient(np.arange(10.), np.arange(10.), w=0)


# ---------------------------------------------------------------------------
# ic_decay
# ---------------------------------------------------------------------------

def test_ic_decay_nan_for_horizon_longer_than_sample():
    prices, fwd = _gbm_panel(T=30, N=8, seed=2)
    factor = fwd.copy()
    decay = ic_decay(factor, prices, horizons=(1, 100))
    assert np.isfinite(decay[0])
    assert np.isnan(decay[1])


# ---------------------------------------------------------------------------
# factor_rank_autocorr
# ---------------------------------------------------------------------------

def test_rank_autocorr_persistent_factor_is_one():
    # A ranking that never changes has a rank autocorrelation of 1.
    rng = np.random.default_rng(6)
    base = rng.standard_normal(30)
    factor = np.tile(base, (100, 1))
    ac = factor_rank_autocorr(factor, lag=1)
    assert np.isnan(ac[0])
    assert np.allclose(ac[1:], 1.0)


def test_rank_autocorr_independent_shuffles_is_zero():
    # Independently reshuffling the cross-section each bar destroys persistence.
    rng = np.random.default_rng(8)
    base = np.arange(40.)
    factor = np.array([rng.permutation(base) for _ in range(300)])
    ac = factor_rank_autocorr(factor, lag=1)
    assert abs(np.nanmean(ac[1:])) < 0.1


def test_rank_autocorr_nan_when_too_few_valid():
    factor = np.array([[1., 2., np.nan, np.nan],
                       [1., 2., 3., 4.]])
    ac = factor_rank_autocorr(factor, lag=1)
    # Bar 1's predecessor (bar 0) has only two finite entries -> NaN.
    assert np.isnan(ac[1])


def test_rank_autocorr_rejects_bad_lag():
    with pytest.raises(ValueError):
        factor_rank_autocorr(np.zeros((5, 5)), lag=0)


# ---------------------------------------------------------------------------
# Top-level exposure
# ---------------------------------------------------------------------------

def test_factor_helpers_exposed_on_top_level_package():
    assert fy.quantile_returns is quantile_returns
    assert fy.roll_information_coefficient is roll_information_coefficient
    assert fy.ic_decay is ic_decay
    assert fy.ic_summary is ic_summary
    assert fy.factor_rank_autocorr is factor_rank_autocorr
    assert fy.QuantileResult is QuantileResult


def test_factor_helpers_not_in_metrics_registry():
    # The two-input factor helpers must not leak into the single-series METRICS
    # registry that drives `summary`.
    from fynance.metrics import METRICS
    for name in ('quantile_returns', 'roll_information_coefficient',
                 'ic_decay', 'ic_summary', 'factor_rank_autocorr'):
        assert name not in METRICS
