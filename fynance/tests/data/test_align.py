#!/usr/bin/env python3
# coding: utf-8

""" Tests for alignment and resampling. """

# Built-in packages
import datetime

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.core import PriceSeries
from fynance.data import align, resample


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


# --- resample ----------------------------------------------------------------

def _weekly_ps():
    # Two calendar weeks (Wed/Thu then next Wed/Thu).
    idx = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-08", "2020-01-09"],
        dtype="datetime64[D]",
    )
    return PriceSeries([1.0, 2.0, 3.0, 4.0], index=idx, name="x")


def test_resample_last_datetime64():
    out = resample(_weekly_ps(), "1w", agg="last")
    assert isinstance(out, PriceSeries)
    assert np.allclose(out.values, [2.0, 4.0])     # last of each week
    assert out.index.dtype.kind == "M"


def test_resample_mean_datetime64():
    out = resample(_weekly_ps(), "1w", agg="mean")
    assert np.allclose(out.values, [1.5, 3.5])


def test_resample_ohlc_datetime64():
    out = resample(_weekly_ps(), "1w", agg="ohlc")
    assert set(out) == {"open", "high", "low", "close"}
    assert np.allclose(out["open"].values, [1.0, 3.0])
    assert np.allclose(out["high"].values, [2.0, 4.0])
    assert np.allclose(out["low"].values, [1.0, 3.0])
    assert np.allclose(out["close"].values, [2.0, 4.0])


def test_resample_unknown_agg_raises():
    with pytest.raises(ValueError):
        resample(_weekly_ps(), "1w", agg="median")


def test_resample_integer_index_raises_clean_error():
    # Default int index: previously a cryptic polars 'every' error.
    ps = PriceSeries([1.0, 2.0, 3.0, 4.0], index=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="datetime64"):
        resample(ps, "1w", agg="last")


def test_resample_object_datetime_index_raises_clean_error():
    # Object-dtype datetime.datetime index: previously an opaque polars error.
    idx = np.array(
        [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)],
        dtype=object,
    )
    ps = PriceSeries([1.0, 2.0], index=idx)
    with pytest.raises(ValueError, match="datetime64"):
        resample(ps, "1w", agg="last")
