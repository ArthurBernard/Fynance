#!/usr/bin/env python3
# coding: utf-8

""" Tests for the CSV adapter. """

# Third-party packages
import numpy as np

# Local packages
from fynance.core import PriceSeries
from fynance.data import load


def test_csv_single_value_column(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("date,close\n2020-01-01,100.0\n2020-01-02,101.0\n2020-01-03,99.5\n")
    ps = load(p)
    assert isinstance(ps, PriceSeries)
    assert ps.name == "close"
    assert ps.values.dtype == np.float64
    assert np.allclose(ps.values, [100.0, 101.0, 99.5])
    assert ps.index.shape == (3,)


def test_csv_multi_value_columns_returns_mapping(tmp_path):
    p = tmp_path / "multi.csv"
    p.write_text("date,a,b\n2020-01-01,1.0,10.0\n2020-01-02,2.0,20.0\n")
    out = load(p)
    assert isinstance(out, dict)
    assert set(out) == {"a", "b"}
    assert np.allclose(out["a"].values, [1.0, 2.0])
    assert np.allclose(out["b"].values, [10.0, 20.0])


def test_csv_no_datetime_index(tmp_path):
    p = tmp_path / "noidx.csv"
    p.write_text("v\n1.0\n2.0\n3.0\n")
    ps = load(p)
    assert isinstance(ps, PriceSeries)
    assert np.allclose(ps.values, [1.0, 2.0, 3.0])
