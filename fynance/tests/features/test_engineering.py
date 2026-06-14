#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.7 feature-engineering tools. """

import numpy as np
import pytest

from fynance.features.engineering import (
    IncrementalMoments,
    granger_causality,
    multi_resolution,
)
from fynance.features.momentums import sma


def test_multi_resolution_shape_and_columns():
    X = np.arange(1.0, 11.0)
    out = multi_resolution(sma, X, [2, 3])
    assert out.shape == (10, 2)
    assert np.allclose(out[:, 0], np.asarray(sma(X, 2)).reshape(-1))
    assert np.allclose(out[:, 1], np.asarray(sma(X, 3)).reshape(-1))


def test_granger_detects_causality():
    rng = np.random.RandomState(0)
    x = rng.standard_normal(300)
    y = np.r_[0.0, 0.8 * x[:-1]] + 0.1 * rng.standard_normal(300)
    _, p = granger_causality(x, y, lag=1)
    assert p < 0.01


def test_granger_independent_not_significant():
    rng = np.random.RandomState(1)
    _, p = granger_causality(rng.standard_normal(300), rng.standard_normal(300), lag=1)
    assert p > 0.05


def test_granger_too_short_raises():
    with pytest.raises(ValueError):
        granger_causality(np.arange(3.0), np.arange(3.0), lag=1)


def test_incremental_moments_matches_numpy():
    rng = np.random.RandomState(2)
    data = rng.standard_normal(100)
    im = IncrementalMoments()
    for v in data:
        im.update(v)
    assert im.n == 100
    assert np.isclose(im.mean, data.mean())
    assert np.isclose(im.var, data.var())
    assert np.isclose(im.std, data.std())


def test_incremental_update_returns_self():
    im = IncrementalMoments()
    assert im.update(1.0) is im
