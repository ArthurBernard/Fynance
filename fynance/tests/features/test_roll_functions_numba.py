#!/usr/bin/env python3
# coding: utf-8

""" roll_min / roll_max numba kernels match the trailing-window spec. """

# Third-party packages
import numpy as np

# Local packages
from fynance.features.roll_functions import roll_max, roll_min


def _ref(X, w, fn):
    T = X.shape[0]
    out = np.empty(T)
    for t in range(T):
        out[t] = fn(X[max(0, t - w + 1): t + 1])
    return out


def test_roll_min_matches_reference():
    rng = np.random.default_rng(7)
    X = rng.normal(10, 2, 40)
    for w in (1, 3, 10, 40):
        assert np.allclose(roll_min(X, w=w, dtype=np.float64), _ref(X, w, np.min))


def test_roll_max_matches_reference():
    rng = np.random.default_rng(8)
    X = rng.normal(10, 2, 40)
    for w in (1, 3, 10, 40):
        assert np.allclose(roll_max(X, w=w, dtype=np.float64), _ref(X, w, np.max))


def test_roll_min_2d():
    rng = np.random.default_rng(9)
    X = rng.normal(10, 2, (30, 3))
    out = roll_min(X, w=5, dtype=np.float64)
    for n in range(3):
        assert np.allclose(out[:, n], _ref(X[:, n], 5, np.min))


def test_roll_min_default_window_is_expanding():
    X = np.array([3.0, 1.0, 2.0, 0.5, 4.0])
    # w=None -> full length -> expanding min
    assert np.allclose(roll_min(X, dtype=np.float64), np.minimum.accumulate(X))


def _multi_col():
    # Time on axis 1, non-square (3, 6).
    return np.array([
        [70., 100., 80., 120., 160., 80.],
        [60., 90., 110., 70., 130., 100.],
        [50., 80., 60., 90., 120., 70.],
    ])


def test_roll_min_axis1_matches_per_row():
    X = _multi_col()
    out = roll_min(X, w=3, axis=1, dtype=np.float64)
    expected = np.vstack([_ref(X[i], 3, np.min) for i in range(X.shape[0])])
    assert np.allclose(out, expected)


def test_roll_max_axis1_matches_per_row():
    X = _multi_col()
    out = roll_max(X, w=3, axis=1, dtype=np.float64)
    expected = np.vstack([_ref(X[i], 3, np.max) for i in range(X.shape[0])])
    assert np.allclose(out, expected)


def test_roll_min_axis1_window_greater_than_n_columns():
    # Window 4 > leading axis length 3 must validate against the time axis.
    X = _multi_col()
    out = roll_min(X, w=4, axis=1, dtype=np.float64)
    expected = np.vstack([_ref(X[i], 4, np.min) for i in range(X.shape[0])])
    assert np.allclose(out, expected)
    assert not np.allclose(
        out, np.vstack([_ref(X[i], 3, np.min) for i in range(X.shape[0])])
    )
