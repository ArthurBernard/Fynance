#!/usr/bin/env python3
# coding: utf-8

""" Test the Information Coefficient / rank-IC metric. """

# Third-party packages
import numpy as np
import pytest

# Local packages
import fynance as fy
from fynance.metrics import information_coefficient


def test_perfect_ic_is_one():
    real = np.array([1., 2., 3., 4., 5., 6.])
    pred = real.copy()
    assert information_coefficient(pred, real, method='spearman') == pytest.approx(1.0)
    assert information_coefficient(pred, real, method='pearson') == pytest.approx(1.0)


def test_inverted_rank_ic_is_minus_one():
    real = np.array([1., 2., 3., 4., 5., 6.])
    pred = -real
    assert information_coefficient(pred, real, method='spearman') == pytest.approx(-1.0)
    assert information_coefficient(pred, real, method='pearson') == pytest.approx(-1.0)


def test_monotone_nonlinear_rank_vs_level():
    # A monotone non-linear map preserves the ordering (rank-IC == 1) but bends
    # the line (pearson < 1): this proves spearman ranks rather than levels.
    real = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
    pred = real ** 3
    spearman = information_coefficient(pred, real, method='spearman')
    pearson = information_coefficient(pred, real, method='pearson')
    assert spearman == pytest.approx(1.0)
    assert pearson < 1.0


def test_pure_noise_ic_is_small():
    rng = np.random.default_rng(0)
    real = rng.standard_normal(2000)
    pred = rng.standard_normal(2000)  # independent of real
    ic = information_coefficient(pred, real, method='spearman')
    assert abs(ic) < 0.1


def test_zero_variance_returns_nan_not_exception():
    real = np.array([1., 2., 3., 4., 5.])
    flat = np.full(5, 7.0)  # zero variance prediction
    assert np.isnan(information_coefficient(flat, real, method='pearson'))
    assert np.isnan(information_coefficient(flat, real, method='spearman'))


def test_fewer_than_two_valid_points_returns_nan():
    # Only one finite pair survives the NaN drop -> nan, no exception.
    pred = np.array([1.0, np.nan, np.nan])
    real = np.array([1.0, 2.0, 3.0])
    assert np.isnan(information_coefficient(pred, real, method='spearman'))
    assert np.isnan(information_coefficient(pred, real, method='pearson'))


def test_nan_pairs_are_dropped():
    # Dropping the NaN pair leaves a perfectly ordered (pred, real) set.
    real = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    pred = np.array([10.0, 20.0, 99.0, 40.0, 50.0])
    assert information_coefficient(pred, real, method='spearman') == pytest.approx(1.0)


def test_panel_cross_sectional_ic_per_bar():
    # Default axis=0 on a (T, N) panel returns one cross-sectional IC per bar.
    pred = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]])
    real = np.array([[1., 2., 3.],    # correct ranking
                     [3., 2., 1.],    # inverted ranking
                     [2., 3., 1.]])   # partial
    ic = information_coefficient(pred, real, method='spearman')
    assert ic.shape == (3,)
    assert ic[0] == pytest.approx(1.0)
    assert ic[1] == pytest.approx(-1.0)
    # Hand-check the third bar: ranks of pred = [1, 2, 3], ranks of real =
    # [2, 3, 1]; spearman == pearson on those ranks.
    from scipy.stats import rankdata
    rp = rankdata(pred[2])
    rr = rankdata(real[2])
    expected = np.corrcoef(rp, rr)[0, 1]
    assert ic[2] == pytest.approx(expected)


def test_panel_axis1_is_time_series_ic_per_asset():
    # axis=1 correlates along time within each column -> one IC per asset.
    real = np.array([[1., 5.],
                     [2., 4.],
                     [3., 3.],
                     [4., 2.],
                     [5., 1.]])
    pred = real.copy()
    ic = information_coefficient(pred, real, method='spearman', axis=1)
    assert ic.shape == (2,)
    assert ic == pytest.approx(np.array([1.0, 1.0]))


def test_unknown_method_raises():
    real = np.array([1., 2., 3.])
    with pytest.raises(ValueError):
        information_coefficient(real, real, method='kendall')


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        information_coefficient(np.array([1., 2.]), np.array([1., 2., 3.]))


def test_exposed_on_top_level_package():
    assert fy.information_coefficient is information_coefficient


def test_ic_decreases_with_noise():
    # IC must decrease monotonically as more noise is mixed into the signal.
    rng = np.random.default_rng(42)
    real = rng.standard_normal(5000)
    noise = rng.standard_normal(5000)
    ics = [
        information_coefficient(real + k * noise, real, method='spearman')
        for k in (0.0, 0.5, 1.0, 2.0, 4.0)
    ]
    assert all(ics[i] > ics[i + 1] for i in range(len(ics) - 1))
    assert ics[0] == pytest.approx(1.0)
