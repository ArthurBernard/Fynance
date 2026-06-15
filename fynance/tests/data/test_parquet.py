#!/usr/bin/env python3
# coding: utf-8

""" Tests for the Parquet adapter (parity with CSV behaviour). """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.core import PriceSeries
from fynance.data import load


def test_parquet_roundtrip(tmp_path):
    pl = pytest.importorskip("polars")
    p = tmp_path / "px.parquet"
    pl.DataFrame({"close": [100.0, 101.0, 99.5]}).write_parquet(p)
    ps = load(p)
    assert isinstance(ps, PriceSeries)
    assert ps.name == "close"
    assert np.allclose(ps.values, [100.0, 101.0, 99.5])
