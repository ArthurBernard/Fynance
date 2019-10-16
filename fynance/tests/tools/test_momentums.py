#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-15 16:35:14
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-15 23:34:52

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
    ma_1d = fy.ema(x_1d, k=4, dtype=np.float32)
    ma_2d = fy.ema(x_2d, k=4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.ema(x_1d, k=7) == fy.ema(x_1d, k=6)).any()
    assert (fy.ema(x_1d, k=7)[1:] != fy.ema(x_1d, k=6)[1:]).all()
    assert (fy.ema(x_2d, k=2, axis=1) == x_2d).all()

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
    ma_1d = fy.emstd(x_1d, k=4, dtype=np.float32)
    ma_2d = fy.emstd(x_2d, k=4, dtype=np.float32)

    assert ma_1d.dtype == np.float32
    assert (ma_1d == ma_2d.flatten()).all()
    assert ma_1d.shape == (6,)
    assert ma_2d.shape == (6, 1)
    assert (fy.emstd(x_2d, k=2, axis=1) == 0.).all()

    a = 1 - 2 / (4 + 1)
    assert (ma_1d == fy.emstd(x_1d, a, dtype=np.float32)).all()
    assert (ma_2d == fy.emstd(x_2d, a, dtype=np.float32)).all()
    assert (fy.emstd(x_1d, 0.) == 0.).all()
    assert (fy.emstd(x_1d, 1.) == 0.).all()
