#!/usr/bin/env python3
# coding: utf-8

""" Tests for :class:`fynance.core.OHLCV`. """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.core import OHLCV


def _synthetic(n=50, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n))
    o = c + rng.standard_normal(n)
    h = np.maximum(o, c) + np.abs(rng.standard_normal(n))
    low = np.minimum(o, c) - np.abs(rng.standard_normal(n))
    v = rng.integers(1, 100, n).astype(float)

    return OHLCV(open=o, high=h, low=low, close=c, volume=v)


def test_construction_and_accessors():
    bars = _synthetic(50)
    assert len(bars) == 50
    assert bars.to_numpy().shape == (50, 5)
    assert bars.columns == ('open', 'high', 'low', 'close', 'volume')
    assert bars.close.dtype == np.float64
    # fixture sanity: high >= low everywhere
    assert np.all(bars.high >= bars.low)


def test_close_required_others_optional():
    bars = OHLCV(close=[1.0, 2.0, 3.0])
    assert len(bars) == 3
    assert bars.columns == ('close',)
    assert bars.has('close') and not bars.has('volume')


def test_absent_field_raises():
    bars = OHLCV(close=[1.0, 2.0], high=[2.0, 3.0])
    with pytest.raises(ValueError, match="no 'volume' field"):
        _ = bars.volume
    with pytest.raises(ValueError, match="no 'open' field"):
        _ = bars.open


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        OHLCV(close=[1.0, 2.0, 3.0], high=[1.0, 2.0])


def test_immutability():
    bars = OHLCV(close=[1.0, 2.0, 3.0])
    assert bars.close.flags.writeable is False
    with pytest.raises(ValueError):
        bars.close[0] = 9.0


def test_to_numpy_columns_follow_order():
    bars = OHLCV(close=[1.0, 2.0], high=[2.0, 3.0])
    # canonical order keeps high before close
    assert bars.columns == ('high', 'close')
    assert np.array_equal(bars.to_numpy()[:, 0], [2.0, 3.0])
    assert np.array_equal(bars.to_numpy()[:, 1], [1.0, 2.0])


def test_from_dict_roundtrip():
    bars = _synthetic(20)
    rebuilt = OHLCV.from_dict(
        {f: getattr(bars, f) for f in bars.columns}
    )
    assert rebuilt == bars


def test_from_dict_requires_close():
    with pytest.raises(ValueError, match="close"):
        OHLCV.from_dict({"high": [1.0, 2.0]})


def test_from_numpy_roundtrip():
    bars = _synthetic(20)
    cols = ('open', 'high', 'low', 'close', 'volume')
    arr = bars.to_numpy()
    rebuilt = OHLCV.from_numpy(arr, columns=cols)
    assert rebuilt == bars


def test_from_numpy_shape_guard():
    with pytest.raises(ValueError, match="incompatible"):
        OHLCV.from_numpy(np.zeros((5, 2)), columns=('open', 'high', 'low'))


def test_equality_distinguishes_fields():
    a = OHLCV(close=[1.0, 2.0], high=[3.0, 4.0])
    b = OHLCV(close=[1.0, 2.0])
    assert a != b
    assert a != "not-an-ohlcv"


def test_repr():
    bars = OHLCV(close=[1.0, 2.0], volume=[10.0, 20.0])
    r = repr(bars)
    assert "len=2" in r and "close" in r and "volume" in r


def test_coercion_from_torch():
    torch = pytest.importorskip("torch")
    c = torch.tensor([1.0, 2.0, 3.0])
    bars = OHLCV(close=c)
    assert np.array_equal(bars.close, np.array([1.0, 2.0, 3.0]))


def test_from_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"Close": [1.0, 2.0], "High": [2.0, 3.0], "x": [9.0, 9.0]})
    bars = OHLCV.from_polars(df)
    assert bars.columns == ('high', 'close')
    assert np.array_equal(bars.high, [2.0, 3.0])
