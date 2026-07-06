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


# -- pandas seams -----------------------------------------------------------

class _DuckFrame:
    """ Minimal pandas-DataFrame look-alike; no pandas import anywhere. """

    class _Col:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float64)

        def to_numpy(self):
            return self._values

    def __init__(self, data):
        self._data = {k: self._Col(v) for k, v in data.items()}
        self.columns = list(data.keys())

    def __getitem__(self, key):
        return self._data[key]


def test_from_pandas_duck_typed_shim_case_insensitive():
    df = _DuckFrame({"Close": [1.0, 2.0], "High": [2.0, 3.0], "x": [9.0, 9.0]})
    bars = OHLCV.from_pandas(df)
    assert bars.columns == ('high', 'close')
    assert np.array_equal(bars.high, [2.0, 3.0])
    assert np.array_equal(bars.close, [1.0, 2.0])


def test_from_pandas_missing_close_raises_naming_it():
    df = _DuckFrame({"open": [1.0, 2.0], "high": [2.0, 3.0]})
    with pytest.raises(ValueError, match="close"):
        OHLCV.from_pandas(df)


def test_from_pandas_missing_volume_is_optional():
    # OHLCV itself treats volume as optional (see class docstring): from_pandas
    # must not raise when the volume column is absent, even though `columns`
    # defaults to naming it.
    df = _DuckFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    bars = OHLCV.from_pandas(df)
    assert not bars.has('volume')
    assert bars.columns == ('open', 'high', 'low', 'close')


def test_from_pandas_missing_open_high_low_also_optional():
    # Same leniency documented for volume applies to open/high/low: only
    # `close` is hard-required, matching the class's own optionality.
    df = _DuckFrame({"close": [1.0, 2.0, 3.0]})
    bars = OHLCV.from_pandas(df)
    assert bars.columns == ('close',)


def test_from_pandas_custom_columns_mapping():
    df = _DuckFrame({
        "Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Vol": [10.0],
    })
    bars = OHLCV.from_pandas(
        df, columns=("Open", "High", "Low", "Close", "Vol")
    )
    assert bars.columns == ('open', 'high', 'low', 'close', 'volume')
    assert np.array_equal(bars.volume, [10.0])


def test_from_pandas_real_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5],
        "close": [1.5, 2.5], "volume": [10.0, 20.0],
    })
    bars = OHLCV.from_pandas(df)
    assert bars.columns == ('open', 'high', 'low', 'close', 'volume')
    assert np.array_equal(bars.close, [1.5, 2.5])
    assert np.array_equal(bars.volume, [10.0, 20.0])


def test_to_pandas_columns_and_dtypes():
    pd = pytest.importorskip("pandas")
    bars = _synthetic(10)
    df = bars.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(bars.columns)
    assert df.shape == (10, 5)
    for col in bars.columns:
        assert df[col].dtype == np.float64
        assert np.array_equal(df[col].to_numpy(), getattr(bars, col))


def test_to_pandas_roundtrip_via_from_pandas():
    pytest.importorskip("pandas")
    bars = _synthetic(15)
    df = bars.to_pandas()
    rebuilt = OHLCV.from_pandas(df)
    assert rebuilt == bars


def test_to_pandas_lazy_import_error_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "pandas":

            raise ImportError("No module named 'pandas'")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="install pandas to use to_pandas"):
        OHLCV(close=[1.0, 2.0]).to_pandas()
