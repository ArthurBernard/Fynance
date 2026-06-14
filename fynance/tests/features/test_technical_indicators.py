#!/usr/bin/env python3
# coding: utf-8

""" Tests for the §5.4 technical indicators (ROC, realized vol, rolling moments). """

# Third-party packages
import numpy as np
import pytest
from scipy import stats as sp_stats

# Local packages
from fynance.features.indicators import (
    realized_volatility,
    roc,
    rolling_autocorr,
    rolling_kurtosis,
    rolling_skewness,
)


@pytest.fixture
def x():
    rng = np.random.RandomState(7)
    return np.cumsum(rng.standard_normal(80)) + 100.0


def test_roc_matches_formula(x):
    w = 5
    out = np.asarray(roc(x, w=w))
    expected = np.zeros_like(x)
    expected[1:w] = (x[1:w] / x[0] - 1) * 100
    expected[w:] = (x[w:] / x[:-w] - 1) * 100
    assert np.allclose(out, expected)


def test_realized_volatility_matches_reference(x):
    w, period = 10, 252
    out = np.asarray(realized_volatility(x, w=w, period=period))
    r = np.log(x[1:] / x[:-1])
    ref = np.zeros_like(x)
    for i in range(len(r)):
        lo = max(0, i - w + 1)
        ref[i + 1] = np.sqrt(period) * np.std(r[lo:i + 1], ddof=0)
    assert np.allclose(out, ref)


def test_rolling_skewness_matches_scipy(x):
    w = 12
    out = np.asarray(rolling_skewness(x, w=w))
    t = 40
    assert np.isclose(out[t], sp_stats.skew(x[t - w + 1:t + 1]))


def test_rolling_kurtosis_matches_scipy(x):
    w = 12
    out = np.asarray(rolling_kurtosis(x, w=w))
    t = 40
    assert np.isclose(out[t], sp_stats.kurtosis(x[t - w + 1:t + 1]))


def test_rolling_autocorr_matches_reference(x):
    w = 15
    out = np.asarray(rolling_autocorr(x, w=w, lag=1))
    t = 50
    win = x[t - w + 1:t + 1]
    a, b = win[:-1], win[1:]
    ref = np.corrcoef(a, b)[0, 1]
    assert np.isclose(out[t], ref)


INDICATORS = [
    ("roc", lambda a: roc(a, w=5)),
    ("realized_volatility", lambda a: realized_volatility(a, w=10)),
    ("rolling_skewness", lambda a: rolling_skewness(a, w=12)),
    ("rolling_kurtosis", lambda a: rolling_kurtosis(a, w=12)),
    ("rolling_autocorr", lambda a: rolling_autocorr(a, w=12)),
]


@pytest.mark.parametrize("name,fn", INDICATORS, ids=[n for n, _ in INDICATORS])
def test_no_lookahead(x, name, fn):
    t = 55
    base = np.asarray(fn(x))
    x2 = x.copy()
    x2[t:] += 50.0
    pert = np.asarray(fn(x2))
    assert np.allclose(base[:t], pert[:t]), f"{name} leaks the future"


@pytest.mark.parametrize("name,fn", INDICATORS, ids=[n for n, _ in INDICATORS])
def test_2d_columnwise(x, name, fn):
    X2 = np.column_stack([x, x[::-1] + 1.0])
    out = np.asarray(fn(X2))
    assert out.shape == X2.shape
    assert np.allclose(out[:, 0], np.asarray(fn(x)))
