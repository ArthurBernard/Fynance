#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.8 robustness metrics (tail_ratio, percent_positive). """

import numpy as np

from fynance.features.stats import percent_positive, tail_ratio


def test_percent_positive():
    X = np.array([0.1, -0.2, 0.3, 0.0, 0.4])
    assert np.isclose(percent_positive(X), 0.6)


def test_percent_positive_2d():
    X = np.array([[0.1, -0.1], [-0.2, 0.2], [0.3, 0.3]])
    assert np.allclose(percent_positive(X), [2 / 3, 2 / 3])


def test_tail_ratio_symmetric_is_one():
    rng = np.random.RandomState(0)
    X = rng.standard_normal(10000)  # symmetric -> tail ratio ~ 1
    assert abs(float(tail_ratio(X)) - 1.0) < 0.1


def test_tail_ratio_matches_quantiles():
    X = np.array([-3., -1., 0., 0., 1., 2., 5.])
    expected = abs(np.quantile(X, 0.95)) / abs(np.quantile(X, 0.05))
    assert np.isclose(tail_ratio(X), expected)
