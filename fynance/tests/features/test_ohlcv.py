#!/usr/bin/env python3
# coding: utf-8

""" Tests for the multi-series OHLCV indicators. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.core import OHLCV
from fynance.features.ohlcv import adx, atr, obv, vwap, williams_r


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    volume = rng.integers(1, 1000, n).astype(float)

    return high, low, close, volume


# -- Golden values (hand-checkable) ---------------------------------------


def test_atr_golden():
    h = np.array([10., 11., 12., 11.5])
    low = np.array([9., 9.5, 11., 10.5])
    c = np.array([9.5, 10.5, 11.5, 11.])
    # TR = [1, 1.5, 1.5, 1]; Wilder RMA w=2 seeded at TR[0]
    assert np.allclose(atr(h, low, c, w=2), [1., 1.25, 1.375, 1.1875])


def test_williams_golden_and_range():
    h = np.array([10., 11., 12., 11.])
    low = np.array([9., 9., 10., 10.])
    c = np.array([9.5, 10.5, 11.5, 10.5])
    out = williams_r(h, low, c, w=2)
    assert np.allclose(out, [-50., -25., -100. / 6., -75.])
    assert np.all((out >= -100.0) & (out <= 0.0))


def test_obv_golden():
    c = np.array([10., 11., 10.5, 10.5, 12.])
    v = np.array([100., 150., 120., 80., 200.])
    assert np.allclose(obv(c, v), [0., 150., 30., 30., 230.])


def test_vwap_cumulative_golden():
    h = np.array([10., 12., 11.])
    low = np.array([8., 10., 9.])
    c = np.array([9., 11., 10.])
    v = np.array([100., 100., 100.])
    assert np.allclose(vwap(h, low, c, v), [9., 10., 10.])


# -- Shapes / ranges ------------------------------------------------------


def test_shapes_match_input():
    h, low, c, v = _synthetic()
    n = h.size
    for out in (atr(h, low, c), adx(h, low, c), williams_r(h, low, c),
                obv(c, v), vwap(h, low, c, v), vwap(h, low, c, v, w=20)):
        assert np.asarray(out).shape == (n,)


def test_atr_nonnegative_and_adx_bounded():
    h, low, c, _ = _synthetic()
    assert np.all(atr(h, low, c) >= 0.0)
    a = adx(h, low, c)
    assert np.all((a >= 0.0) & (a <= 100.0))


def test_williams_in_range():
    h, low, c, _ = _synthetic()
    out = williams_r(h, low, c)
    assert np.all((out >= -100.0) & (out <= 0.0))


# -- OHLCV dispatch -------------------------------------------------------


def test_ohlcv_dispatch_matches_raw():
    h, low, c, v = _synthetic()
    bars = OHLCV(high=h, low=low, close=c, volume=v)
    assert np.allclose(atr(bars), atr(h, low, c))
    assert np.allclose(adx(bars), adx(h, low, c))
    assert np.allclose(williams_r(bars), williams_r(h, low, c))
    assert np.allclose(obv(bars), obv(c, v))
    assert np.allclose(vwap(bars), vwap(h, low, c, v))


def test_missing_args_raise():
    with pytest.raises(ValueError, match="high, low and close"):
        atr(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="close and volume"):
        obv(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="high, low, close and volume"):
        vwap(np.array([1.0, 2.0]))


# -- No-lookahead ---------------------------------------------------------


def test_no_lookahead():
    h, low, c, v = _synthetic(n=200)
    t = 120
    for fn, args in (
        (atr, (h, low, c)),
        (adx, (h, low, c)),
        (williams_r, (h, low, c)),
        (obv, (c, v)),
        (lambda *a: vwap(*a, w=20), (h, low, c, v)),
    ):
        full = np.asarray(fn(*args))
        truncated = np.asarray(fn(*(a[:t] for a in args)))
        assert np.allclose(full[:t], truncated, equal_nan=True), fn


# -- Parity against a slow pure-python reference --------------------------


def _ref_williams(h, low, c, w):
    out = np.empty(h.size)
    for t in range(h.size):
        a = max(0, t - w + 1)
        hh = h[a:t + 1].max()
        ll = low[a:t + 1].min()
        d = hh - ll
        out[t] = 0.0 if d == 0 else -100.0 * (hh - c[t]) / d

    return out


def _ref_obv(c, v):
    out = np.zeros(c.size)
    for t in range(1, c.size):
        if c[t] > c[t - 1]:
            out[t] = out[t - 1] + v[t]
        elif c[t] < c[t - 1]:
            out[t] = out[t - 1] - v[t]
        else:
            out[t] = out[t - 1]

    return out


def test_parity_with_reference():
    h, low, c, v = _synthetic(n=200, seed=3)
    assert np.allclose(williams_r(h, low, c, w=14), _ref_williams(h, low, c, 14),
                       atol=1e-9)
    assert np.allclose(obv(c, v), _ref_obv(c, v), atol=1e-9)
