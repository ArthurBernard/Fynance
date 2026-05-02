#!/usr/bin/env python3
# coding: utf-8

""" Benchmarks for Kalman filter functions. """

# Built-in
# (none)

# Third-party
import numpy as np
import pytest

# Local
from fynance.features.filters import kalman_filter, rts_smoother


@pytest.fixture
def kalman_data():
    """Generate test data for Kalman filter."""
    rng = np.random.default_rng(42)
    T, n = 1000, 2
    y = rng.standard_normal((T, n))
    G = np.eye(n)
    F = np.eye(n)
    W = np.eye(n) * 0.1
    V = np.eye(n) * 0.5
    return y, G, F, W, V


@pytest.mark.benchmark(group="kalman_filter")
def test_kalman_filter_1000_steps(benchmark, kalman_data):
    """Benchmark kalman_filter over 1000 time steps."""
    y, G, F, W, V = kalman_data
    result = benchmark(kalman_filter, y, G, F, W, V)
    assert len(result) == 6
    m, C, a, R, e, S = result
    assert m.shape == (1000, 2)


@pytest.mark.benchmark(group="rts_smoother")
def test_rts_smoother_1000_steps(benchmark, kalman_data):
    """Benchmark rts_smoother over 1000 time steps."""
    y, G, F, W, V = kalman_data
    m, C, a, R, e, S = kalman_filter(y, G, F, W, V)
    result = benchmark(rts_smoother, m, C, a, R, F)
    assert len(result) == 2
    ms, Cs = result
    assert ms.shape == (1000, 2)
