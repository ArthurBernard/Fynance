#!/usr/bin/env python3
# coding: utf-8

""" Multi-series OHLCV value object.

Defines :class:`OHLCV`, a thin numpy-backed container for **aligned**
Open/High/Low/Close/Volume series. It is the multi-series counterpart of
:class:`~fynance.core.price_series.PriceSeries` and the input contract the
multi-series technical indicators (ATR, ADX, Williams %R, OBV, VWAP) consume.

Design invariants
-----------------
- **Composition, never subclassing** of :class:`numpy.ndarray`: each present
  field is stored as a contiguous, read-only ``float64`` array.
- **Aligned & validated**: every present field shares the same length; a
  mismatch raises at construction.
- **Sparse by design**: ``close`` is required, the other four fields are
  optional. Accessing an absent field raises a clear error rather than guessing.

"""

from __future__ import annotations

# Built-in packages
from collections.abc import Mapping
from typing import Any

# Third-party packages
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ['OHLCV']

#: Canonical field order for stacking / iteration.
_FIELDS = ('open', 'high', 'low', 'close', 'volume')


def _coerce_1d(values: ArrayLike) -> NDArray[np.float64]:
    """ Coerce an array-like to a read-only 1-D ``float64`` array.

    Mirrors the coercion used by :class:`~fynance.core.price_series.PriceSeries`
    (numpy / torch / polars array-likes are accepted via ``np.asarray``).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    v.flags.writeable = False

    return v


class OHLCV:
    """ Thin, numpy-backed container of aligned OHLCV series.

    Holds up to five aligned 1-D ``float64`` arrays — ``open``, ``high``,
    ``low``, ``close``, ``volume`` — sharing a common length. ``close`` is
    required; the others are optional and raise an informative error when
    accessed while absent. Composes numpy arrays rather than subclassing them.

    Parameters
    ----------
    close : array-like
        Close series (required, defines the length).
    open, high, low, volume : array-like, optional
        Other OHLCV fields; each, when given, must match ``close`` in length.

    Attributes
    ----------
    open, high, low, close, volume : numpy.ndarray
        Read-only ``float64`` 1-D arrays. Accessing an absent field raises
        :class:`ValueError`.

    Examples
    --------
    >>> bars = OHLCV(close=[10., 11., 12.], high=[10.5, 11.5, 12.5],
    ...              low=[9.5, 10.5, 11.5])
    >>> len(bars)
    3
    >>> bars.high
    array([10.5, 11.5, 12.5])
    >>> bars.columns
    ('high', 'low', 'close')
    >>> bars.volume
    Traceback (most recent call last):
        ...
    ValueError: OHLCV has no 'volume' field

    """

    def __init__(
        self,
        close: ArrayLike,
        open: ArrayLike | None = None,   # noqa: A002  (OHLCV domain name)
        high: ArrayLike | None = None,
        low: ArrayLike | None = None,
        volume: ArrayLike | None = None,
    ):
        """ Initialize the container, coercing and length-checking each field. """
        raw = {
            'open': open, 'high': high, 'low': low,
            'close': close, 'volume': volume,
        }
        self._data: dict[str, NDArray[np.float64]] = {}

        n: int | None = None
        for field in _FIELDS:
            arr = raw[field]

            if arr is None:

                continue

            coerced = _coerce_1d(arr)

            if n is None:
                n = coerced.size

            elif coerced.size != n:

                raise ValueError(
                    f"field {field!r} length {coerced.size} != length {n}"
                )

            self._data[field] = coerced

        self._n = int(n) if n is not None else 0

    # -- Constructors -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, ArrayLike]) -> OHLCV:
        """ Build from a mapping of field name -> array-like.

        Only the canonical keys (``open``/``high``/``low``/``close``/``volume``)
        are read; ``close`` must be present.

        Examples
        --------
        >>> OHLCV.from_dict({"close": [1., 2.], "volume": [10., 20.]}).columns
        ('close', 'volume')

        """
        if 'close' not in data:

            raise ValueError("from_dict requires at least a 'close' field")

        kwargs = {f: data[f] for f in _FIELDS if f in data}

        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_numpy(
        cls,
        array: NDArray,
        columns: tuple[str, ...] = _FIELDS,
    ) -> OHLCV:
        """ Build from a 2-D array whose columns are named by ``columns``.

        Parameters
        ----------
        array : numpy.ndarray
            Shape ``(N, k)`` with ``k == len(columns)``.
        columns : tuple of str
            Field name of each column, in order. Defaults to the full
            ``(open, high, low, close, volume)``.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[9., 11., 8., 10.], [10., 12., 9., 11.]])
        >>> OHLCV.from_numpy(a, columns=("open", "high", "low", "close")).columns
        ('open', 'high', 'low', 'close')

        """
        a = np.asarray(array, dtype=np.float64)

        if a.ndim != 2 or a.shape[1] != len(columns):

            raise ValueError(
                f"array shape {a.shape} incompatible with columns {columns}"
            )

        return cls.from_dict({c: a[:, i] for i, c in enumerate(columns)})

    @classmethod
    def from_polars(cls, data: Any) -> OHLCV:
        """ Build from a polars DataFrame (columns matched case-insensitively).

        Any column whose lower-cased name is one of ``open``/``high``/``low``/
        ``close``/``volume`` is mapped to the matching field.
        """
        import polars as pl

        if not isinstance(data, pl.DataFrame):

            raise TypeError(f"unsupported polars input: {type(data)!r}")

        mapping = {}
        for col in data.columns:
            key = col.lower()

            if key in _FIELDS:
                mapping[key] = data[col].to_numpy()

        return cls.from_dict(mapping)

    # -- Field access -----------------------------------------------------

    def _get(self, field: str) -> NDArray[np.float64]:
        """ Return a present field or raise an informative error. """
        try:

            return self._data[field]

        except KeyError:

            raise ValueError(f"OHLCV has no {field!r} field") from None

    @property
    def open(self) -> NDArray[np.float64]:    # noqa: A003
        """ Open series (raises if absent). """
        return self._get('open')

    @property
    def high(self) -> NDArray[np.float64]:
        """ High series (raises if absent). """
        return self._get('high')

    @property
    def low(self) -> NDArray[np.float64]:
        """ Low series (raises if absent). """
        return self._get('low')

    @property
    def close(self) -> NDArray[np.float64]:
        """ Close series (always present). """
        return self._get('close')

    @property
    def volume(self) -> NDArray[np.float64]:
        """ Volume series (raises if absent). """
        return self._get('volume')

    @property
    def columns(self) -> tuple[str, ...]:
        """ Present fields, in canonical OHLCV order. """
        return tuple(f for f in _FIELDS if f in self._data)

    def has(self, field: str) -> bool:
        """ Whether ``field`` is present. """
        return field in self._data

    # -- Bridges ----------------------------------------------------------

    def to_numpy(self) -> NDArray[np.float64]:
        """ Column-stack the present fields into a ``(N, k)`` array.

        Columns follow :attr:`columns` (canonical OHLCV order).

        Examples
        --------
        >>> OHLCV(close=[1., 2., 3.], high=[2., 3., 4.]).to_numpy().shape
        (3, 2)

        """
        if not self._data:

            return np.empty((0, 0), dtype=np.float64)

        return np.column_stack([self._data[f] for f in self.columns])

    # -- Dunders ----------------------------------------------------------

    def __len__(self) -> int:
        """ Number of bars. """
        return self._n

    def __eq__(self, other: object) -> bool:
        """ Equality by present fields and their values. """
        if not isinstance(other, OHLCV):

            return NotImplemented

        if self.columns != other.columns:

            return False

        return all(
            np.array_equal(self._data[f], other._data[f], equal_nan=True)
            for f in self.columns
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """ Short representation: length and present fields. """
        return f"OHLCV(len={self._n}, fields={list(self.columns)})"
