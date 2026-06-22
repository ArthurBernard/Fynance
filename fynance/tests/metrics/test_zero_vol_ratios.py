#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" sharpe/sortino must not crash on a zero-volatility (constant) curve. """

# Third-party
import numpy as np

# Local
from fynance.metrics import calmar, roll_calmar, roll_sharpe, sharpe, sortino
from fynance.metrics.ratios import _safe_ratio
from fynance.metrics.summary import summary


def test_flat_curve_is_zero_not_crash():
    flat = np.full(20, 100.0)
    assert sharpe(flat) == 0.0
    assert sortino(flat) == 0.0


def test_safe_ratio_zero_denominator():
    # Riskless gain -> +inf; riskless loss -> -inf; flat (0/0) -> 0; mixed
    # array handled element-wise.
    assert np.isposinf(_safe_ratio(1.0, 0.0))
    assert np.isneginf(_safe_ratio(-1.0, 0.0))
    assert _safe_ratio(0.0, 0.0) == 0.0
    r = _safe_ratio(np.array([1.0, -1.0, 0.0, 2.0]),
                    np.array([0.0, 0.0, 0.0, 2.0]))
    assert np.isposinf(r[0])
    assert np.isneginf(r[1])
    assert r[2] == 0.0
    assert r[3] == 1.0


def test_flat_curve_with_rf_is_riskless_loss():
    # A flat (zero-volatility) curve evaluated against a positive risk-free
    # rate is a guaranteed underperformance: the ratio limit is -inf, not the
    # best-possible +inf. This guards the sign bug in _safe_ratio.
    flat = np.full(20, 100.0)
    assert np.isneginf(sharpe(flat, rf=0.05))
    assert np.isneginf(sortino(flat, rf=0.05))


def test_2d_zero_vol_negative_excess_is_neg_inf():
    # 2-D array path: a flat column with a positive rf scores -inf (riskless
    # loss), while a real column stays finite.
    flat = np.full(50, 100.0)
    rng = np.random.default_rng(0)
    normal = 100.0 * np.cumprod(1 + rng.standard_normal(50) * 0.01)
    X = np.column_stack([flat, normal])

    res = sharpe(X, rf=0.05, axis=0)
    assert res.shape == (2,)
    assert np.isneginf(res[0])
    assert np.isfinite(res[1])


def test_roll_sharpe_zero_vol_is_signed_inf_not_zero():
    # A zero-rolling-vol window must follow the calmar/roll_calmar +/-inf
    # convention, not the old 0.0 zeroing. A flat curve has zero return over
    # zero volatility at every window: a negative risk-free rate makes the
    # excess positive (+inf), a positive one makes it a riskless loss (-inf).
    flat = np.full(6, 100.0)
    assert np.all(np.isposinf(roll_sharpe(flat, rf=-0.05, period=12)))
    assert np.all(np.isneginf(roll_sharpe(flat, rf=0.05, period=12)))


def test_2d_mixed_columns_no_crash():
    rng = np.random.default_rng(0)
    normal = 100.0 * np.cumprod(1 + rng.standard_normal(50) * 0.01)
    flat = np.full(50, 100.0)
    X = np.column_stack([flat, normal])

    res = sharpe(X, axis=0)
    assert res.shape == (2,)
    assert res[0] == 0.0
    assert np.isfinite(res[1])


def test_summary_on_flat_curve():
    # The regression that surfaced the bug: summary() over a flat equity curve.
    out = summary(np.full(30, 100.0))
    assert out["sharpe"] == 0.0
    assert np.isfinite(out["max_drawdown"])


def test_calmar_zero_mdd_is_inf_not_zero():
    # A profitable, drawdown-free curve has zero maximum drawdown. Calmar must
    # follow the same zero-denominator convention as sharpe/sortino (+inf for a
    # riskless gain), not return 0.0 (the worst possible Calmar).
    up = np.array([100., 101., 102., 103., 104., 105.])
    assert np.isposinf(calmar(up, period=12))
    # a flat curve (zero return over zero drawdown) stays 0.0
    assert calmar(np.full(10, 100.0), period=12) == 0.0


def test_calmar_zero_mdd_2d_columns():
    up = np.array([100., 101., 102., 103., 104., 105.])
    drawdown = np.array([100., 90., 95., 80., 85., 70.])
    X = np.column_stack([up, drawdown])
    res = calmar(X, period=12)
    assert res.shape == (2,)
    assert np.isposinf(res[0])      # drawdown-free profitable column
    assert np.isfinite(res[1])      # column with a real drawdown


def test_roll_calmar_zero_mdd_is_inf_not_zero():
    # Same convention for the rolling variant: a window that has gained without
    # any drawdown yet must score +inf, not 0.0.
    X = np.array([70., 100., 80., 120., 160., 80.])
    rc = roll_calmar(X, period=12)
    # index 1 = strictly increasing so far (70 -> 100), zero MDD, positive ret.
    assert np.isposinf(rc[1])
    assert rc[0] == 0.0             # first point: zero return over zero MDD
