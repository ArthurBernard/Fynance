#!/usr/bin/env python3
# coding: utf-8

""" Property tests for the rolling feature kernels.

Two invariants that the per-function unit tests do not check directly:

1. **Reference parity** — each Numba kernel matches an independent,
   pure-NumPy reference of the same definition. This is the cross-check that
   would catch a window off-by-one.
2. **No lookahead** — every kernel is strictly causal: output ``[:t]`` is
   unchanged when the future ``X[t:]`` is perturbed.
"""

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.features.momentums import ema, sma, smstd, wma
from fynance.features.roll_functions import roll_max, roll_min

# --------------------------------------------------------------------------- #
# Independent NumPy references (expanding for the first w-1 points, then a
# trailing window of size w — the convention fynance uses).
# --------------------------------------------------------------------------- #

def _expanding_trailing(x, w, fn):
    out = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = fn(x[lo:i + 1])
    return out


def ref_sma(x, w):
    return _expanding_trailing(x, w, np.mean)


def ref_smstd(x, w):
    return _expanding_trailing(x, w, lambda a: np.std(a, ddof=0))


def ref_wma(x, w):
    def f(win):
        wt = np.arange(1, len(win) + 1)
        return float((win * wt).sum() / wt.sum())
    return _expanding_trailing(x, w, f)


def ref_roll_min(x, w):
    return _expanding_trailing(x, w, np.min)


def ref_roll_max(x, w):
    return _expanding_trailing(x, w, np.max)


def ref_ema(x, alpha):
    out = np.empty(len(x), dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * x[i]
    return out


@pytest.fixture
def x():
    rng = np.random.RandomState(123)
    return np.cumsum(rng.randn(60)) + 50.0


# --------------------------------------------------------------------------- #
# 1. Reference parity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("w", [2, 3, 5, 13])
def test_sma_matches_reference(x, w):
    assert np.allclose(np.asarray(sma(x, w=w)), ref_sma(x, w))


@pytest.mark.parametrize("w", [2, 3, 5, 13])
def test_wma_matches_reference(x, w):
    assert np.allclose(np.asarray(wma(x, w=w)), ref_wma(x, w))


@pytest.mark.parametrize("w", [2, 3, 5, 13])
def test_smstd_matches_reference(x, w):
    assert np.allclose(np.asarray(smstd(x, w=w)), ref_smstd(x, w))


@pytest.mark.parametrize("w", [2, 3, 5, 13])
def test_roll_min_matches_reference(x, w):
    assert np.allclose(np.asarray(roll_min(x, w=w)), ref_roll_min(x, w))


@pytest.mark.parametrize("w", [2, 3, 5, 13])
def test_roll_max_matches_reference(x, w):
    assert np.allclose(np.asarray(roll_max(x, w=w)), ref_roll_max(x, w))


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.8, 0.94])
def test_ema_matches_reference(x, alpha):
    assert np.allclose(np.asarray(ema(x, alpha=alpha)), ref_ema(x, alpha))


# --------------------------------------------------------------------------- #
# 2. No lookahead (strict causality)
# --------------------------------------------------------------------------- #

CAUSAL_FUNCS = [
    ("sma", lambda a: sma(a, w=5)),
    ("wma", lambda a: wma(a, w=5)),
    ("ema", lambda a: ema(a, alpha=0.8)),
    ("smstd", lambda a: smstd(a, w=5)),
    ("roll_min", lambda a: roll_min(a, w=5)),
    ("roll_max", lambda a: roll_max(a, w=5)),
]


@pytest.mark.parametrize("name,func", CAUSAL_FUNCS, ids=[n for n, _ in CAUSAL_FUNCS])
def test_no_lookahead(x, name, func):
    t = 40
    base = np.asarray(func(x))
    x_future = x.copy()
    x_future[t:] += 1000.0  # corrupt the future only
    perturbed = np.asarray(func(x_future))
    # outputs strictly before t must not see the future
    assert np.allclose(base[:t], perturbed[:t]), f"{name} leaks future data"
