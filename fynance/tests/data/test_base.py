#!/usr/bin/env python3
# coding: utf-8

""" Tests for :func:`fynance.data.base.frame_to_series`. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.core import PriceSeries
from fynance.data.base import frame_to_series


def test_frame_to_series_explicit_value_col():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "a": [1.0, 2.0],
            "b": [10.0, 20.0],
        }
    ).with_columns(pl.col("date").str.to_date())
    # Explicit value_col selects a single column even though several exist.
    out = frame_to_series(df, value_col="b")
    assert isinstance(out, PriceSeries)
    assert out.name == "b"
    assert np.allclose(out.values, [10.0, 20.0])
    # the temporal column is resolved as the index
    assert out.index.dtype.kind == "M"


def test_frame_to_series_explicit_value_col_no_index():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    out = frame_to_series(df, value_col="a")
    assert isinstance(out, PriceSeries)
    assert out.name == "a"
    assert np.allclose(out.values, [1.0, 2.0, 3.0])


def test_frame_to_series_multi_column_mapping():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    out = frame_to_series(df)
    assert isinstance(out, dict)
    assert set(out) == {"a", "b"}
