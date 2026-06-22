#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-11 20:05:31
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-19 11:44:53

""" Test scale functions. """

# Built-in packages

# Third party packages
import numpy as np
import pytest

# Local packages
import fynance as fy


@pytest.fixture()
def set_variables():
    x_1d = np.array([60, 100, 80, 120, 160, 80])
    x_2d = x_1d.reshape([6, 1])
    return x_1d, x_2d


def test_scale_std(set_variables):
    x_1d, x_2d = set_variables
    s = fy.Scale(x_1d, kind="std", a=1, b=2)
    assert s.params["m"] == 100
    assert s.params["s"] == 32.65986323710904
    assert (s(x_1d) == 2 * (x_1d - 100) / 32.65986323710904 + 1).all()
    assert (s.revert(s(x_1d)) == x_1d).all()
    s = fy.Scale(x_2d, kind="std", a=1, b=2)
    assert s.params["m"] == 100
    assert s.params["s"] == 32.65986323710904
    assert (s(x_2d) == 2 * (x_2d - 100) / 32.65986323710904 + 1).all()
    assert (s.revert(s(x_2d)) == x_2d).all()


def test_scale_norm(set_variables):
    x_1d, x_2d = set_variables
    s = fy.Scale(x_1d, kind="norm", a=1, b=3)
    assert s.params["m"] == 60
    assert s.params["s"] == 160
    assert (s(x_1d) == 2 * (x_1d - 60) / 100 + 1).all()
    assert (s.revert(s(x_1d)) == x_1d).all()
    s = fy.Scale(x_2d, kind="norm", a=1, b=3)
    assert s.params["m"] == 60
    assert s.params["s"] == 160
    assert (s(x_2d) == 2 * (x_2d - 60) / 100 + 1).all()
    assert (s.revert(s(x_2d)) == x_2d).all()


def test_scale_roll_std(set_variables):
    x_1d, x_2d = set_variables
    w = 3
    s = fy.Scale(x_1d, w=w, kind="roll_std", a=1, b=2, kind_moment="s")
    mean = fy.sma(x_1d, w)
    std = fy.smstd(x_1d, w)
    std[std == 0.] = 1.
    scaled = (x_1d - mean) / std
    assert (s.params["m"] == mean).all()
    assert (s.params["s"] == std).all()
    assert (s(x_1d) == 2 * scaled + 1).all()
    assert (s.revert(s(x_1d)) == x_1d).all()

    s = fy.Scale(x_2d, w=w, kind="roll_std", a=1, b=2, kind_moment="e")
    mean = fy.ema(x_2d, w=w, axis=0)
    std = fy.emstd(x_2d, w=w, axis=0)
    std[std == 0.] = 1.
    scaled = (x_2d - mean) / std
    assert (s.params["m"] == mean).all()
    assert (s.params["s"] == std).all()
    assert (s(x_2d) == 2 * scaled + 1).all()
    assert (s.revert(s(x_2d)) == x_2d).all()

    s = fy.Scale(x_2d, w=w, axis=1, kind="roll_std", a=1, b=2, kind_moment="w")
    mean = fy.wma(x_2d, w=w, axis=1)
    std = fy.wmstd(x_2d, w=w, axis=1)
    std[std == 0.] = 1.
    scaled = (x_2d - mean) / std
    assert (s.params["m"] == mean).all()
    assert (s.params["s"] == std).all()
    assert (s(x_2d) == (2 * scaled + 1).T).all()
    assert (s.revert(s(x_2d)) == x_2d).all()


def test_scale_roll_norm(set_variables):
    x_1d, x_2d = set_variables
    w = 3
    s = fy.Scale(x_1d, w=w, kind="roll_norm", a=1, b=2)
    mean = fy.roll_min(x_1d, w)
    std = fy.roll_max(x_1d, w)
    idx = std == mean
    mean[idx] = 0.
    std[idx] *= 2
    scaled = (x_1d - mean) / (std - mean)
    assert (s.params["m"] == mean).all()
    assert (s.params["s"] == std).all()
    assert (s(x_1d) == (2 - 1) * scaled + 1).all()
    assert (s.revert(s(x_1d)) == x_1d).all()

    s = fy.Scale(x_2d, w=w, axis=1, kind="roll_norm", a=1, b=2)
    mean = fy.roll_min(x_2d, w=w, axis=1)
    std = fy.roll_max(x_2d, w=w, axis=1)
    idx = std == mean
    mean[idx] = 0.
    std[idx] *= 2
    scaled = (x_2d - mean) / (std - mean)
    assert (s.params["m"] == mean).all()
    assert (s.params["s"] == std).all()
    assert (s(x_2d) == ((2 - 1) * scaled + 1).T).all()
    assert (s.revert(s(x_2d)) == x_2d).all()


# §5.5 rank-based normalization

def test_roll_rank_values():
    import numpy as np

    from fynance.features.scale import roll_rank
    X = np.array([1., 3., 2., 5., 4.])
    assert np.allclose(roll_rank(X, w=3), [0.5, 1., 0.5, 1., 0.5])


def test_roll_rank_in_unit_interval():
    import numpy as np

    from fynance.features.scale import roll_rank
    rng = np.random.RandomState(0)
    out = np.asarray(roll_rank(rng.standard_normal(200), w=20))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_roll_rank_no_lookahead():
    import numpy as np

    from fynance.features.scale import roll_rank
    rng = np.random.RandomState(1)
    X = rng.standard_normal(100)
    t = 60
    base = np.asarray(roll_rank(X, w=20))
    X2 = X.copy()
    X2[t:] += 10.0
    pert = np.asarray(roll_rank(X2, w=20))
    assert np.allclose(base[:t], pert[:t])


def test_roll_rank_2d_columnwise():
    import numpy as np

    from fynance.features.scale import roll_rank
    X = np.array([1., 3., 2., 5., 4.])
    X2 = np.column_stack([X, X[::-1]])
    out = np.asarray(roll_rank(X2, w=3))
    assert out.shape == (5, 2)
    assert np.allclose(out[:, 0], roll_rank(X, w=3))


# --------------------------------------------------------------------------- #
#   Genuine multi-column axis=1 parity (roadmap 1.A)                           #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def multi_col():
    # Time on axis 1, distinct rows, non-square (3, 4).
    return np.array([
        [1., 2., 3., 4.],
        [6., 5., 7., 8.],
        [2., 9., 1., 3.],
    ])


@pytest.mark.parametrize(
    "kind, base_func",
    [("std", "standardize"), ("norm", "normalize")],
)
def test_scale_axis1_matches_per_row(multi_col, kind, base_func):
    # Scale(axis=1) must equal stacking the 1-D transform of each row, and
    # must round-trip through revert. Previously the axis=1 branch discarded
    # its result (missing return) and silently fell back to axis=0 behaviour.
    f = getattr(fy, base_func)
    s = fy.Scale(multi_col, kind=kind, axis=1, a=0, b=1)
    out = s.scale(multi_col)
    expected = np.vstack([f(multi_col[i]) for i in range(multi_col.shape[0])])
    assert np.allclose(out, expected)
    assert np.allclose(s.revert(out), multi_col)


@pytest.mark.parametrize("kind", ["std", "norm"])
def test_scale_axis1_differs_from_axis0(multi_col, kind):
    s1 = fy.Scale(multi_col, kind=kind, axis=1, a=0, b=1)
    s0 = fy.Scale(multi_col, kind=kind, axis=0, a=0, b=1)
    assert not np.allclose(s1.scale(multi_col), s0.scale(multi_col))


# --------------------------------------------------------------------------- #
#   Standalone rolling functions: genuine multi-column axis=1 parity          #
#   (roadmap 1.2 — PR #193 fixed the Scale path but not these functions)      #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def roll_multi_col():
    # Time on axis 1, distinct rows, non-square (3, 6); no constant rolling
    # window so the rolling std/range never collapses to zero.
    return np.array([
        [1., 3., 2., 5., 4., 6.],
        [6., 4., 7., 2., 8., 3.],
        [2., 9., 1., 4., 7., 5.],
    ])


@pytest.mark.parametrize("kind_moment", ["s", "w", "e"])
def test_roll_standardize_axis1_matches_per_row(roll_multi_col, kind_moment):
    # The standalone roll_standardize used to raise on genuine multi-column
    # data with axis=1: its rolling-moment parameters stayed in input
    # orientation (rows, cols) while X.T was (cols, rows), so the broadcast
    # failed. The result must now equal stacking the 1-D transform of each row.
    from fynance.features.scale import roll_standardize
    out = roll_standardize(roll_multi_col, w=3, axis=1, kind_moment=kind_moment)
    expected = np.vstack([
        roll_standardize(roll_multi_col[i], w=3, kind_moment=kind_moment)
        for i in range(roll_multi_col.shape[0])
    ])
    assert np.allclose(out, expected, equal_nan=True)


def test_roll_normalize_axis1_matches_per_row(roll_multi_col):
    # Same broadcast bug as roll_standardize, on the rolling min/max path.
    from fynance.features.scale import roll_normalize
    out = roll_normalize(roll_multi_col, w=3, axis=1)
    expected = np.vstack([
        roll_normalize(roll_multi_col[i], w=3)
        for i in range(roll_multi_col.shape[0])
    ])
    assert np.allclose(out, expected, equal_nan=True)


@pytest.mark.parametrize("func", ["roll_standardize", "roll_normalize"])
def test_standalone_roll_axis1_differs_from_axis0(roll_multi_col, func):
    # Guard against silently falling back to the axis=0 result.
    from fynance.features import scale
    f = getattr(scale, func)
    out1 = f(roll_multi_col, w=3, axis=1)
    out0 = f(roll_multi_col, w=3, axis=0)
    assert not np.allclose(out1, out0, equal_nan=True)
