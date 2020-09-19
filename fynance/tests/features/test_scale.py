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
