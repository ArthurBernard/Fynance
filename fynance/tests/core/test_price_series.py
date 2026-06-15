#!/usr/bin/env python3
# coding: utf-8

""" Tests for :class:`fynance.core.PriceSeries`. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.core import PriceSeries


def test_construction_and_len():
    ps = PriceSeries([100.0, 101.0, 99.0], name="px", freq="1d")
    assert len(ps) == 3
    assert ps.name == "px"
    assert ps.freq == "1d"
    assert ps.values.dtype == np.float64
    # default index is 0..n-1
    assert np.array_equal(ps.index, np.arange(3))


def test_immutability():
    ps = PriceSeries([1.0, 2.0, 3.0])
    assert ps.values.flags.writeable is False
    with pytest.raises(ValueError):
        ps.values[0] = 9.0
    # derived ops return new objects
    assert ps.to_returns() is not ps


def test_numpy_interop():
    ps = PriceSeries([1.0, 2.0, 3.0])
    arr = np.asarray(ps)
    assert np.array_equal(arr, [1.0, 2.0, 3.0])
    assert float(np.sum(ps)) == 6.0


def test_getitem_scalar_and_slice():
    ps = PriceSeries([10.0, 20.0, 30.0, 40.0], index=[5, 6, 7, 8])
    assert ps[1] == 20.0
    sub = ps[1:3]
    assert isinstance(sub, PriceSeries)
    assert np.array_equal(sub.values, [20.0, 30.0])
    assert np.array_equal(sub.index, [6, 7])


def test_index_length_mismatch():
    with pytest.raises(ValueError):
        PriceSeries([1.0, 2.0], index=[1, 2, 3])


def test_equality_and_repr():
    a = PriceSeries([1.0, 2.0, 3.0])
    b = PriceSeries([1.0, 2.0, 3.0])
    c = PriceSeries([1.0, 2.0, 4.0])
    assert a == b
    assert a != c
    assert "PriceSeries(" in repr(a)


def test_to_returns_pct_log_raw():
    ps = PriceSeries([100.0, 110.0, 99.0])
    assert np.allclose(ps.to_returns("pct").values, [0.1, -0.1])
    assert np.allclose(ps.to_returns("raw").values, [10.0, -11.0])
    log_r = ps.to_returns("log").values
    assert np.allclose(log_r, np.log([110.0 / 100.0, 99.0 / 110.0]))


def test_to_returns_dropna_false_keeps_length():
    ps = PriceSeries([100.0, 110.0, 99.0])
    r = ps.to_returns("pct", dropna=False)
    assert len(r) == 3
    assert np.isnan(r.values[0])


def test_returns_prices_roundtrip():
    rng = np.random.default_rng(0)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 50))
    ps = PriceSeries(prices)
    for kind in ("pct", "log", "raw"):
        r = ps.to_returns(kind, dropna=True)
        rebuilt = r.to_prices(base=prices[0], kind=kind)
        assert np.allclose(rebuilt.values, prices, atol=1e-9)


def test_cumulative():
    r = PriceSeries([0.1, -0.1, 0.05])
    eq = r.cumulative()
    assert np.allclose(eq.values, np.cumprod([1.1, 0.9, 1.05]))


def test_pnl_is_causal():
    # returns and a position book
    returns = PriceSeries([0.01, 0.02, -0.03, 0.04])
    positions = [1.0, 1.0, -1.0, -1.0]
    pnl = returns.pnl(positions)
    # first is NaN (no prior position)
    assert np.isnan(pnl.values[0])
    # pnl_t = position_{t-1} * r_t
    assert np.isclose(pnl.values[1], 1.0 * 0.02)
    assert np.isclose(pnl.values[2], 1.0 * -0.03)
    assert np.isclose(pnl.values[3], -1.0 * 0.04)


def test_pnl_no_lookahead():
    # changing a future position must not change earlier pnl
    returns = PriceSeries([0.01, 0.02, -0.03, 0.04])
    base = returns.pnl([1.0, 1.0, 1.0, 1.0]).values
    perturbed = returns.pnl([1.0, 1.0, 1.0, -5.0]).values
    assert np.allclose(base[:3], perturbed[:3], equal_nan=True)


def test_dropna_fillna():
    ps = PriceSeries([np.nan, 1.0, np.nan, 2.0])
    assert np.array_equal(ps.drop_na().values, [1.0, 2.0])
    assert np.array_equal(ps.fillna(0.0).values, [0.0, 1.0, 0.0, 2.0])


def test_to_numpy_and_pipe():
    ps = PriceSeries([1.0, 2.0, 3.0])
    assert np.array_equal(ps.to_numpy(), [1.0, 2.0, 3.0])

    # length-preserving -> wrapped back
    doubled = ps.pipe(lambda a: a * 2)
    assert isinstance(doubled, PriceSeries)
    assert np.array_equal(doubled.values, [2.0, 4.0, 6.0])

    # reducing -> raw result
    total = ps.pipe(np.sum)
    assert float(total) == 6.0


def test_to_torch_lazy():
    torch = pytest.importorskip("torch")
    ps = PriceSeries([1.0, 2.0, 3.0])
    t = ps.to_torch()
    assert t.dtype == torch.float32
    assert t.shape == (3,)


def test_from_polars_series_and_frame():
    pl = pytest.importorskip("polars")
    s = pl.Series("px", [1.0, 2.0, 3.0])
    ps = PriceSeries.from_polars(s)
    assert np.array_equal(ps.values, [1.0, 2.0, 3.0])
    assert ps.name == "px"

    df = pl.DataFrame({"px": [10.0, 11.0, 12.0]})
    ps2 = PriceSeries.from_polars(df, value_col="px")
    assert np.array_equal(ps2.values, [10.0, 11.0, 12.0])
