#!/usr/bin/env python3
# coding: utf-8

""" Tests for the iso-vol money-management coefficient. """

# Third-party packages
import numpy as np

# Local packages
from fynance.features.momentums import ema
from fynance.features.money_management import iso_vol


def test_iso_vol_uses_standard_returns():
    # iso_vol must use the standard return s_t / s_{t-1} - 1, not the reciprocal
    # s_{t-1} / s_t - 1 that the function used before the fix.
    series = np.array([95, 100, 85, 105, 110, 90]).astype(np.float64)
    target_vol, leverage, period, half_life = 0.5, 2.0, 12, 3

    out = iso_vol(series, target_vol=target_vol, leverage=leverage,
                  period=period, half_life=half_life)

    ret2 = np.square(series[1:] / series[:-1] - 1)
    vol = np.sqrt(period * ema(ret2, w=half_life))
    vol[vol <= 0.0] = 1e-8
    expected = np.ones(series.size)
    expected[2:] = target_vol / vol[:-1]
    expected[expected > leverage] = leverage

    assert np.allclose(out, expected)


def test_iso_vol_causality():
    # iv[t] must depend only on series[:t+1] (in fact series[:t]): truncating the
    # series at t+1 leaves iv[t] unchanged -- no future leakage into the signal.
    rng = np.random.default_rng(0)
    series = 100 * np.cumprod(1 + rng.standard_normal(60) * 0.02)
    full = iso_vol(series, target_vol=0.2, leverage=3.0, period=252, half_life=11)

    for t in range(2, series.size):
        trunc = iso_vol(series[:t + 1], target_vol=0.2, leverage=3.0,
                        period=252, half_life=11)
        assert np.isclose(trunc[t], full[t]), (t, trunc[t], full[t])


def test_iso_vol_leverage_cap():
    # A near-flat (ultra-low-vol) series would demand huge leverage; the output
    # must be capped at `leverage`.
    flat = np.full(20, 100.0)
    leverage = 2.5
    out = iso_vol(flat, target_vol=0.2, leverage=leverage)

    assert np.all(out <= leverage + 1e-12)
    assert out.max() == leverage


def test_iso_vol_volatility_floor():
    # The 1e-8 floor on `vol` keeps the coefficient finite (no division by zero)
    # even when the series has zero realized volatility.
    flat = np.full(10, 50.0)
    out = iso_vol(flat, target_vol=0.2, leverage=1.5)

    assert np.all(np.isfinite(out))
    assert not np.any(np.isnan(out))


def test_iso_vol_warmup_is_one():
    # The first two coefficients are seeded to 1 (no return / vol yet).
    rng = np.random.default_rng(1)
    series = 100 * np.cumprod(1 + rng.standard_normal(30) * 0.01)
    out = iso_vol(series)

    assert out[0] == 1.0
    assert out[1] == 1.0


def test_iso_vol_accepts_list_input():
    # A plain list/tuple used to raise a bare AttributeError ('list' object has
    # no attribute 'size'); the input is now coerced like in sibling modules.
    data = [95.0, 100.0, 85.0, 105.0, 110.0, 90.0]
    out_list = iso_vol(data, target_vol=0.5, leverage=2, period=12, half_life=3)
    out_arr = iso_vol(np.asarray(data), target_vol=0.5, leverage=2,
                      period=12, half_life=3)

    assert isinstance(out_list, np.ndarray)
    assert np.allclose(out_list, out_arr)
    # Tuples work too.
    assert np.allclose(iso_vol(tuple(data), target_vol=0.5, leverage=2,
                               period=12, half_life=3), out_arr)
