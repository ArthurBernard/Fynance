#!/usr/bin/env python3
# coding: utf-8

""" Tests for alignment and resampling. """

# Third-party packages
import numpy as np

# Local packages
from fynance.core import PriceSeries
from fynance.data import align


def test_outer_align_ffill_is_causal():
    a = PriceSeries([1.0, 2.0, 3.0], index=[1, 2, 3], name="a")
    b = PriceSeries([10.0, 30.0], index=[1, 3], name="b")
    out = align({"a": a, "b": b}, how="outer", fill="ffill")
    assert np.array_equal(out["a"].index, [1, 2, 3])
    # b has no value at t=2 -> forward-filled from t=1 (past only)
    assert np.allclose(out["b"].values, [10.0, 10.0, 30.0])


def test_outer_align_no_fill_keeps_nan():
    a = PriceSeries([1.0, 2.0, 3.0], index=[1, 2, 3], name="a")
    b = PriceSeries([10.0, 30.0], index=[1, 3], name="b")
    out = align({"a": a, "b": b}, how="outer", fill=None)
    assert np.isnan(out["b"].values[1])


def test_inner_align_intersection():
    a = PriceSeries([1.0, 2.0, 3.0], index=[1, 2, 3], name="a")
    b = PriceSeries([10.0, 30.0], index=[1, 3], name="b")
    out = align({"a": a, "b": b}, how="inner")
    assert np.array_equal(out["a"].index, [1, 3])
    assert np.allclose(out["a"].values, [1.0, 3.0])
    assert np.allclose(out["b"].values, [10.0, 30.0])


def test_ffill_leading_nan_stays_nan():
    a = PriceSeries([1.0, 2.0, 3.0], index=[1, 2, 3], name="a")
    b = PriceSeries([30.0], index=[3], name="b")
    out = align({"a": a, "b": b}, how="outer", fill="ffill")
    # no past value before t=3 -> leading NaNs
    assert np.isnan(out["b"].values[0])
    assert np.isnan(out["b"].values[1])
    assert out["b"].values[2] == 30.0
