#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-11 20:05:31
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-11 20:20:24

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
