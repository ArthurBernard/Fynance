#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-15 16:35:14
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-30 13:47:52

""" Test momentum functions. """

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


def test_sma(set_variables):
    x_1d, x_2d = set_variables
    ma_1d = fy.sma(x_1d, 4, dtype=np.float32)
    ma_2d = fy.sma(x_2d, 4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.sma(x_1d, 7) == fy.sma(x_1d, 6)).all()
    assert (fy.sma(x_2d, 2, axis=1) == x_2d).all()


def test_wma(set_variables):
    x_1d, x_2d = set_variables
    ma_1d = fy.wma(x_1d, 4, dtype=np.float32)
    ma_2d = fy.wma(x_2d, 4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.wma(x_1d, 7) == fy.wma(x_1d, 6)).all()
    assert (fy.wma(x_2d, 2, axis=1) == x_2d).all()


def test_ema(set_variables):
    x_1d, x_2d = set_variables
    ma_1d = fy.ema(x_1d, w=4, dtype=np.float32)
    ma_2d = fy.ema(x_2d, w=4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.ema(x_1d, w=7) == fy.ema(x_1d, w=6)).any()
    assert (fy.ema(x_1d, w=7)[1:] != fy.ema(x_1d, w=6)[1:]).all()
    assert (fy.ema(x_2d, w=2, axis=1) == x_2d).all()

    a = 1 - 2 / (4 + 1)
    assert (ma_1d == fy.ema(x_1d, a, dtype=np.float32)).all()
    assert (ma_2d == fy.ema(x_2d, a, dtype=np.float32)).all()
    assert (fy.ema(x_1d, 0.) == x_1d).all()
    assert (fy.ema(x_1d, 1.) == x_1d[0]).all()


def test_smstd(set_variables):
    x_1d, x_2d = set_variables
    ma_1d = fy.smstd(x_1d, 4, dtype=np.float32)
    ma_2d = fy.smstd(x_2d, 4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.smstd(x_1d, 7) == fy.smstd(x_1d, 6)).all()
    assert (fy.smstd(x_2d, 2, axis=1) == 0.).all()

    for t in range(1, 6):
        s1 = fy.smstd(x_1d, ddof=0, w=0, dtype=np.float64)
        s2 = np.std(x_1d[:t + 1], ddof=0)
        assert s1[t] == pytest.approx(s2)
        s1 = fy.smstd(x_1d, ddof=0, w=3, dtype=np.float64)
        s2 = np.std(x_1d[max(0, t - 3 + 1): t + 1], ddof=0)
        assert s1[t] == pytest.approx(s2)
        if t >= 2:
            s1 = fy.smstd(x_1d, ddof=2, w=0, dtype=np.float64)
            s2 = np.std(x_1d[:t + 1], ddof=2)
            assert s1[t] == pytest.approx(s2)
            s1 = fy.smstd(x_1d, ddof=2, w=3, dtype=np.float64)
            s2 = np.std(x_1d[max(0, t - 3 + 1): t + 1], ddof=2)
            assert s1[t] == pytest.approx(s2)

    with pytest.raises(ValueError) as execinfo:
        fy.smstd(x_1d, ddof=2, w=2, dtype=np.float64)
    execinfo.match(r'w=2.*ddof=2')

    with pytest.raises(ValueError) as execinfo:
        fy.smstd(x_1d, ddof=3, w=2, dtype=np.float64)
    execinfo.match(r'w=2.*ddof=3')


def test_wmstd(set_variables):
    x_1d, x_2d = set_variables
    ma_1d = fy.wmstd(x_1d, 4, dtype=np.float32)
    ma_2d = fy.wmstd(x_2d, 4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.wmstd(x_1d, 7) == fy.wmstd(x_1d, 6)).all()
    assert (fy.wmstd(x_2d, 2, axis=1) == 0.).all()


def test_emstd(set_variables):
    x_1d, x_2d = set_variables
    ma_1d = fy.emstd(x_1d, w=4, dtype=np.float32)
    ma_2d = fy.emstd(x_2d, w=4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.emstd(x_2d, w=2, axis=1) == 0.).all()

    a = 1 - 2 / (4 + 1)
    assert (ma_1d == fy.emstd(x_1d, a, dtype=np.float32)).all()
    assert (ma_2d == fy.emstd(x_2d, a, dtype=np.float32)).all()
    assert (fy.emstd(x_1d, 0.) == 0.).all()
    assert (fy.emstd(x_1d, 1.) == 0.).all()


# --------------------------------------------------------------------------- #
#   Genuine multi-column axis=1 parity (roadmap 1.A)                           #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def multi_col():
    # Time on axis 1, three distinct rows, non-square (3, 6).
    return np.array([
        [70., 100., 80., 120., 160., 80.],
        [60., 90., 110., 70., 130., 100.],
        [50., 80., 60., 90., 120., 70.],
    ])


def _stack_rows(f, X, **kwargs):
    return np.vstack([f(X[i], **kwargs) for i in range(X.shape[0])])


@pytest.mark.parametrize("name", ["sma", "wma", "smstd", "wmstd"])
def test_moving_axis1_matches_per_row(multi_col, name):
    # axis=1 must equal stacking the 1-D result of each row.
    f = getattr(fy, name)
    out = f(multi_col, w=3, axis=1)
    assert np.allclose(out, _stack_rows(f, multi_col, w=3))


@pytest.mark.parametrize("name", ["sma", "wma", "smstd", "wmstd"])
def test_moving_axis1_window_greater_than_n_columns(multi_col, name):
    # The window must be validated against the time axis (axis=1, length 6),
    # not the leading axis (length 3). A window of 4 > 3 used to be silently
    # clamped to 3 before the axis transpose.
    f = getattr(fy, name)
    out = f(multi_col, w=4, axis=1)
    assert np.allclose(out, _stack_rows(f, multi_col, w=4))
    # axis=1 with w=4 differs from the (buggy) clamped w=3 result.
    assert not np.allclose(out, _stack_rows(f, multi_col, w=3))


def test_sma_axis1_differs_from_axis0():
    # On a non-square multi-column array, axis=1 and axis=0 disagree.
    X = np.array([
        [1., 2., 3., 4.],
        [5., 7., 9., 11.],
        [2., 4., 8., 16.],
    ])
    assert not np.allclose(fy.sma(X, w=2, axis=1), fy.sma(X, w=2, axis=0))


@pytest.mark.parametrize(
    "name", ["sma", "wma", "smstd", "wmstd", "ema", "emstd"],
)
def test_axis_error_on_1d(name):
    # axis=1 on a 1-D array must raise a clean AxisError (np.AxisError was
    # removed in NumPy 2; the size check also used to raise IndexError first).
    f = getattr(fy, name)
    with pytest.raises(np.exceptions.AxisError):
        f(np.arange(5.), axis=1)


def test_z_score_axis1_matches_per_row(multi_col):
    from fynance.features.stats import z_score
    for kind in ("s", "w", "e"):
        out = z_score(multi_col, w=3, kind=kind, axis=1)
        expected = np.array([
            z_score(multi_col[i], w=3, kind=kind)
            for i in range(multi_col.shape[0])
        ])
        assert np.allclose(out, expected)


def test_roll_z_score_axis1_matches_per_row(multi_col):
    from fynance.features.stats import roll_z_score
    for kind in ("s", "w", "e"):
        out = roll_z_score(multi_col, w=3, kind=kind, axis=1)
        expected = _stack_rows(roll_z_score, multi_col, w=3, kind=kind)
        assert np.allclose(out, expected)


def test_roll_z_score_e_kind_single_conversion(multi_col):
    # The e->alpha conversion w = 1 - 2/(1+w) must be applied exactly once,
    # regardless of axis: axis=1 must equal the per-row 1-D result.
    from fynance.features.stats import roll_z_score
    out = roll_z_score(multi_col, w=3, kind="e", axis=1)
    expected = _stack_rows(roll_z_score, multi_col, w=3, kind="e")
    assert np.allclose(out, expected)


def test_mad_axis1_matches_per_row():
    from fynance.features.stats import mad
    # Non-square array whose leading axis differs from the trailing axis so a
    # broadcast bug (X.T - mean) would raise instead of returning per-row mad.
    X = np.array([
        [1., 2.],
        [3., 4.],
        [5., 6.],
    ])
    out = mad(X, axis=1)
    expected = np.array([mad(X[i]) for i in range(X.shape[0])])
    assert np.allclose(out, expected)
    # axis=0 stays correct (per-column).
    assert np.allclose(
        mad(X, axis=0), [mad(X[:, j]) for j in range(X.shape[1])]
    )
