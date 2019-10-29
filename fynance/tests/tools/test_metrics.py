#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-23 12:31:27
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-29 12:08:33

""" Test metric functions. """

# Built-in packages

# Third party packages
import numpy as np
import pytest

# Local packages
import fynance as fy
from fynance._exceptions import ArraySizeError


@pytest.fixture()
def set_variables():
    x_1d = np.array([60, 100, 80, 120, 160, 80])
    x_2d = x_1d.reshape([6, 1])
    return x_1d, x_2d


def test_accuracy(set_variables):
    x_1d, x_2d = set_variables
    f = fy.accuracy

    assert f(x_1d, x_1d + 1, sign=True) == 1
    assert f(x_1d, x_1d + 1, sign=False) == 0
    assert f(x_2d, x_2d + 1, sign=True) == np.array([1])
    assert f(x_2d, x_2d + 1, sign=False) == np.array([0])


def test_annual_return(set_variables):
    x_1d, x_2d = set_variables
    f = fy.annual_return
    a_1d = f(x_1d, period=12, dtype=np.float32)
    a_2d = f(x_2d, period=12, dtype=np.float32)

    assert a_1d.dtype == np.float32
    assert (a_1d == a_2d.flatten()).all()
    assert a_1d.shape == ()
    assert a_2d.shape == (1,)

    # test axis wrapper
    with pytest.raises(ArraySizeError) as execinfo:
        f(x_2d, axis=1)
    execinfo.match(r'1 .* 1 .* 2')

    # test ddof wrapper
    with pytest.raises(ArraySizeError) as execinfo:
        f(x_2d, axis=0, ddof=7)
    execinfo.match(r'7.*6')


def test_annual_volatility(set_variables):
    x_1d, x_2d = set_variables
    f = fy.annual_volatility
    a_1d = f(x_1d, period=12, dtype=np.float32)
    a_2d = f(x_2d, period=12, dtype=np.float32)

    assert a_1d.dtype == np.float32
    assert (a_1d == a_2d.flatten()).all()
    assert a_1d.shape == ()
    assert a_2d.shape == (1,)

    with pytest.raises(ArraySizeError) as execinfo:
        f(x_2d, axis=1)
    execinfo.match(r'1 .* 1 .* 2')


def test_calmar(set_variables):
    x_1d, x_2d = set_variables
    f = fy.calmar
    a_1d = f(x_1d, period=12, dtype=np.float32)
    a_2d = f(x_2d, period=12, dtype=np.float32)

    assert a_1d.dtype == np.float32
    assert (a_1d == a_2d.flatten()).all()
    assert a_1d.shape == ()
    assert a_2d.shape == (1,)

    with pytest.raises(ArraySizeError) as execinfo:
        f(x_2d, axis=1)
    execinfo.match(r'1 .* 1 .* 2')


def test_diversified_ratio():
    # TODO: test
    pass


def test_drawdown(set_variables):
    x_1d, x_2d = set_variables
    f = fy.drawdown
    a_1d = f(x_1d, dtype=np.float32)
    a_2d = f(x_2d, dtype=np.float32)

    assert a_1d.dtype == np.float32
    assert (a_1d == a_2d.flatten()).all()
    assert a_1d.shape == (6,)
    assert a_2d.shape == (6, 1)

    assert (f(x_2d, axis=1, raw=True) == 0).all()
    assert (f(x_2d, axis=1, dtype=np.float32) == 0).all()
    assert (f(x_1d, raw=True) == np.array([0, 0, 20, 0, 0, 80])).all()
    assert (f(x_2d, raw=True).flatten() == f(x_1d, raw=True)).all()
    res = np.array([[0, 0], [0,  0], [0, 80]])
    assert (f(x_2d.reshape([3, 2]), axis=1, raw=True, dtype=np.float32) == res).all()


def test_mad(set_variables):
    x_1d, x_2d = set_variables
    f = fy.mad
    a_1d = f(x_1d, dtype=np.float32)
    a_2d = f(x_2d, dtype=np.float32)

    assert a_1d.dtype == np.float32
    assert (a_1d == a_2d.flatten()).all()
    assert a_1d.shape == ()
    assert a_2d.shape == (1,)

    assert (f(x_2d, axis=1) == 0).all()
    res = np.array([20, 20, 40])
    assert (f(x_2d.reshape([3, 2]), axis=1) == res).all()


def test_mdd(set_variables):
    x_1d, x_2d = set_variables
    f = fy.mdd
    a_1d = f(x_1d, dtype=np.float32)
    a_2d = f(x_2d, dtype=np.float32)

    assert a_1d.dtype == np.float32
    assert (a_1d == a_2d.flatten()).all()
    assert a_1d.shape == ()
    assert a_2d.shape == (1,)

    assert (f(x_2d, axis=1, raw=True) == 0).all()
    assert (f(x_2d, axis=1, dtype=np.float32) == 0).all()
    assert f(x_1d, raw=True) == 80
    assert f(x_2d, raw=True).flatten() == f(x_1d, raw=True)
    res = np.array([0, 0, 80])
    assert (f(x_2d.reshape([3, 2]), axis=1, raw=True, dtype=np.float32) == res).all()



def test_perf_index():
    # TODO : test
    pass


def test_perf_returns():
    # TODO : test
    pass


def test_perf_strat():
    # TODO : test
    pass


def test_sharpe(set_variables):
    x_1d, x_2d = set_variables
    f = fy.sharpe
    #a_1d = f(x_1d, dtype=np.float32)
    #a_2d = f(x_2d, dtype=np.float32)

    #assert a_1d.dtype == np.float32
    #assert (a_1d == a_2d.flatten()).all()
    #assert a_1d.shape == ()
    #assert a_2d.shape == (1,)

    with pytest.raises(ArraySizeError) as execinfo:
        f(x_2d, axis=1)
    execinfo.match(r'1 .* 1 .* 2')

    with pytest.raises(ArraySizeError) as execinfo:
        f(x_2d, axis=0, ddof=7)
    execinfo.match(r'7.*6')

    res = np.array([2.8867514, -0.4330127], dtype=np.float32)
    #assert (f(x_2d.reshape([3, 2]), period=3, dtype=np.float32) == res).all()


def test_zscore():
    pass


def test_roll_annual_return(set_variables):
    x_1d, x_2d = set_variables
    f = fy.annual_return
    roll_f = fy.roll_annual_return
    a_1d = roll_f(x_1d, period=12, dtype=np.float32)
    for t in range(1, x_1d.size):
        assert a_1d[t] == f(x_1d[: t + 1], period=12, dtype=np.float32)
