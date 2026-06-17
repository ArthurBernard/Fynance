#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" sharpe/sortino must not crash on a zero-volatility (constant) curve. """

# Third-party
import numpy as np

# Local
from fynance.metrics import sharpe, sortino
from fynance.metrics.ratios import _safe_ratio
from fynance.metrics.summary import summary


def test_flat_curve_is_zero_not_crash():
    flat = np.full(20, 100.0)
    assert sharpe(flat) == 0.0
    assert sortino(flat) == 0.0


def test_safe_ratio_zero_denominator():
    # Riskless gain -> +inf; flat (0/0) -> 0; mixed array handled element-wise.
    assert np.isposinf(_safe_ratio(1.0, 0.0))
    assert _safe_ratio(0.0, 0.0) == 0.0
    r = _safe_ratio(np.array([1.0, 0.0, 2.0]), np.array([0.0, 0.0, 2.0]))
    assert np.isposinf(r[0]) and r[1] == 0.0 and r[2] == 1.0


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
