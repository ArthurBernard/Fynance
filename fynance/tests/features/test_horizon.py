#!/usr/bin/env python3
# coding: utf-8

""" Test forward horizon-return labels. """

# Third-party packages
import numpy as np
import pytest

# Local packages
import fynance as fy
from fynance.features import horizon_returns


def test_non_overlapping_length_and_values():
    # Constant 10% step => every 2-step forward return is 21%.
    prices = 100. * 1.1 ** np.arange(7)  # length 7
    out = horizon_returns(prices, 2)
    # base indices 0, 2, 4 (t + h < T = 7) => length floor((7 - 1) / 2) = 3.
    assert out.shape == (3,)
    expected = prices[np.array([2, 4, 6])] / prices[np.array([0, 2, 4])] - 1.
    assert out == pytest.approx(expected)
    assert out == pytest.approx(np.full(3, 1.1 ** 2 - 1.))


def test_overlapping_length_and_values():
    prices = 100. * 1.1 ** np.arange(7)
    out = horizon_returns(prices, 2, overlapping=True)
    # base indices 0..4 => length T - h = 5.
    assert out.shape == (5,)
    base = np.arange(5)
    expected = prices[base + 2] / prices[base] - 1.
    assert out == pytest.approx(expected)


def test_horizon_one_matches_simple_returns():
    prices = np.array([100., 110., 99., 108.9, 130.68])
    out = horizon_returns(prices, 1, overlapping=True)
    expected = prices[1:] / prices[:-1] - 1.
    assert out == pytest.approx(expected)


def test_manual_computation():
    prices = np.array([10., 12., 9., 18., 20., 15.])
    out = horizon_returns(prices, 3)
    # Non-overlapping: base indices 0, 3 (t + 3 < 6) => length floor(5 / 3) = 1.
    # Only base index 0 keeps t + 3 = 3 < 6; index 3 would need t + 3 = 6.
    assert out.shape == (1,)
    assert out[0] == pytest.approx(18. / 10. - 1.)


def test_leakage_probe_beyond_window_does_not_change_label():
    # The label at t depends ONLY on the endpoints prices[t] and prices[t+h].
    # Perturbing any price that is not an endpoint of label 0's window must not
    # change label 0 (no leakage); perturbing label 1's forward endpoint must.
    rng = np.random.default_rng(0)
    prices = 100. * np.cumprod(1. + rng.standard_normal(20) * 0.01)
    h = 4
    base = horizon_returns(prices, h)  # base indices 0, 4, 8, 12 -> endpoints 4, 8, 12, 16
    # Perturb prices[16], strictly beyond label 0's forward endpoint (index 4).
    perturbed = prices.copy()
    perturbed[16] *= 2.0
    after = horizon_returns(perturbed, h)
    # Label 0 (uses prices[0], prices[4]) is unchanged.
    assert after[0] == pytest.approx(base[0])
    # Label 3 (base index 12 -> uses prices[12], prices[16]) changes,
    # confirming the probe actually moved an endpoint.
    assert after[3] != pytest.approx(base[3])


def test_panel_shape():
    rng = np.random.default_rng(1)
    prices = 100. * np.cumprod(1. + rng.standard_normal((30, 4)) * 0.01, axis=0)
    out = horizon_returns(prices, 5)
    # base indices 0, 5, ..., 25 (t + 5 < 30) => length floor(29 / 5) = 5.
    assert out.shape == (5, 4)
    base_idx = np.arange(0, 30 - 5, 5)
    expected = prices[base_idx + 5] / prices[base_idx] - 1.
    assert out == pytest.approx(expected)


def test_strictly_causal_label_uses_future_only():
    # Label at t equals prices[t+h]/prices[t]-1; it must not depend on prices
    # strictly between t and t+h.
    prices = np.array([1., 2., 4., 8., 16., 32.])
    h = 3
    out = horizon_returns(prices, h, overlapping=True)
    # First label (t=0) = prices[3]/prices[0]-1 = 8/1-1 = 7.
    assert out[0] == pytest.approx(7.0)
    perturbed = prices.copy()
    perturbed[1] = 999.  # strictly inside the (0, 3) window
    perturbed[2] = -999.
    out2 = horizon_returns(perturbed, h, overlapping=True)
    assert out2[0] == pytest.approx(out[0])


def test_invalid_horizon_raises():
    prices = np.array([1., 2., 3.])
    with pytest.raises(ValueError):
        horizon_returns(prices, 0)
    with pytest.raises(ValueError):
        horizon_returns(prices, -1)


def test_horizon_too_large_raises():
    prices = np.array([1., 2., 3.])
    with pytest.raises(ValueError):
        horizon_returns(prices, 3)  # T == h, no label fits


def test_exposed_on_top_level_package():
    assert fy.horizon_returns is horizon_returns
