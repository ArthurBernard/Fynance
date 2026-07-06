#!/usr/bin/env python3
# coding: utf-8

""" Central financial time-series value object.

Defines :class:`PriceSeries`, a thin, numpy-backed container for a 1-D
financial series with its temporal index and metadata. It is the spine of
the :mod:`fynance` 2.x pipeline (data -> features -> signal -> portfolio ->
backtest -> metrics).

Design invariants
-----------------
- **Composition, never subclassing** of :class:`numpy.ndarray`: the values
  are stored as a contiguous, read-only ``float64`` array; transformations
  return *new* :class:`PriceSeries` objects.
- **numpy is the lingua franca**: pytorch is never stored, only produced on
  demand through :meth:`PriceSeries.to_torch`.
- **Thin object**: only the fundamental price/return identities live here;
  the rich transform library stays as free functions reachable through
  :meth:`PriceSeries.pipe`.

"""

from __future__ import annotations

# Built-in packages
from typing import Any, Callable

# Third-party packages
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ['PriceSeries']


class PriceSeries:
    """ Thin, numpy-backed financial time-series.

    Wraps a 1-D array of values together with a temporal ``index`` and light
    metadata (``name``, ``freq``). It composes a numpy array rather than
    subclassing it, which keeps numpy interop (``np.asarray(ps)`` works)
    without the fragility of ndarray subclassing.

    Parameters
    ----------
    values : array-like
        1-D sequence of values (coerced to ``float64``).
    index : array-like, optional
        Temporal index of the same length. Defaults to ``0..n-1`` (int64).
    name : str, optional
        Series name.
    freq : str, optional
        Sampling frequency tag (e.g. ``"1d"``), purely informational.

    Attributes
    ----------
    values : numpy.ndarray
        Read-only ``float64`` 1-D array of values.
    index : numpy.ndarray
        1-D array aligned with ``values``.
    name : str or None
        Series name.
    freq : str or None
        Frequency tag.

    Examples
    --------
    >>> ps = PriceSeries([100., 101., 99.], name="px")
    >>> len(ps)
    3
    >>> ps[1]
    101.0

    """

    def __init__(
        self,
        values: ArrayLike,
        index: ArrayLike | None = None,
        name: str | None = None,
        freq: str | None = None,
    ):
        """ Initialize the series. """
        v = np.asarray(values, dtype=np.float64).reshape(-1).copy()
        v.flags.writeable = False
        self.values: NDArray[np.float64] = v

        if index is None:
            idx = np.arange(v.size, dtype=np.int64)

        else:
            idx = np.asarray(index).reshape(-1)

            if idx.size != v.size:

                raise ValueError(
                    f"index length {idx.size} != values length {v.size}"
                )

        idx.flags.writeable = False
        self.index: NDArray[Any] = idx
        self.name = name
        self.freq = freq

    # -- Constructors -----------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        values: NDArray,
        index: NDArray | None = None,
        name: str | None = None,
        freq: str | None = None,
    ) -> PriceSeries:
        """ Build a :class:`PriceSeries` from numpy arrays. """
        return cls(values, index=index, name=name, freq=freq)

    @classmethod
    def from_polars(
        cls,
        data: Any,
        value_col: str | None = None,
        index_col: str | None = None,
        freq: str | None = None,
    ) -> PriceSeries:
        """ Build a :class:`PriceSeries` from a polars Series or DataFrame.

        A :class:`polars.Series` maps directly to the values. For a
        :class:`polars.DataFrame`, ``value_col`` selects the value column
        (defaults to the single non-index numeric column) and ``index_col``
        (or the first temporal column) becomes the index.
        """
        import polars as pl

        if isinstance(data, pl.Series):

            return cls(
                data.to_numpy(),
                name=value_col or data.name,
                freq=freq,
            )

        if not isinstance(data, pl.DataFrame):

            raise TypeError(f"unsupported polars input: {type(data)!r}")

        # Resolve the index column: explicit, else first temporal column.
        if index_col is None:
            temporal = [
                c for c, dt in zip(data.columns, data.dtypes)
                if dt in (pl.Date, pl.Datetime)
            ]
            index_col = temporal[0] if temporal else None

        index = data[index_col].to_numpy() if index_col is not None else None

        if value_col is None:
            # Match the documented default: the single non-index *numeric*
            # column. Filtering by name alone would pick a lone string column
            # and fail later with an opaque cast-to-float64 error.
            candidates = [
                c for c, dt in zip(data.columns, data.dtypes)
                if c != index_col and dt.is_numeric()
            ]

            if len(candidates) != 1:

                raise ValueError(
                    "value_col is ambiguous; pass it explicitly "
                    f"(numeric candidates={candidates})"
                )

            value_col = candidates[0]

        return cls(
            data[value_col].to_numpy(),
            index=index,
            name=value_col,
            freq=freq,
        )

    @classmethod
    def from_pandas(cls, obj: Any, freq: str | None = None) -> PriceSeries:
        """ Build a :class:`PriceSeries` from a duck-typed pandas-like object.

        Accepts anything exposing ``.to_numpy()`` or ``.values`` -- a
        :class:`pandas.Series`, a single-column :class:`pandas.DataFrame`, or
        a minimal look-alike. No pandas import happens in this method: it is
        pure duck typing, so any object satisfying the protocol works (this is
        what lets the test suite exercise it with a pandas-free shim).

        Parameters
        ----------
        obj : Any
            Duck-typed pandas-like object (``Series`` or single-column
            ``DataFrame``).
        freq : str, optional
            Frequency tag carried onto the series.

        Returns
        -------
        PriceSeries

        Raises
        ------
        TypeError
            If ``obj`` exposes neither ``.to_numpy()`` nor ``.values``.
        ValueError
            If the extracted array is 2-D with more than one column (an
            ambiguous, multi-column ``DataFrame``).

        Notes
        -----
        Unlike :meth:`from_polars` (which drops the index -- polars frames
        rarely carry a meaningful one), :class:`PriceSeries` is built here
        with its ``index`` attribute populated from ``obj.index`` whenever
        that attribute is present (e.g. a pandas ``Series``'s ``DatetimeIndex``),
        converted through :func:`numpy.asarray`. It is only left at the
        default ``0..n-1`` range when ``obj`` exposes no ``.index`` at all,
        as with the minimal duck-typed shim used in the tests.

        Examples
        --------
        >>> import pandas as pd
        >>> s = pd.Series([100., 101., 99.], name="px")
        >>> PriceSeries.from_pandas(s).values
        array([100., 101.,  99.])

        """
        if hasattr(obj, "to_numpy"):
            values = obj.to_numpy()

        elif hasattr(obj, "values"):
            values = obj.values

        else:

            raise TypeError(
                "from_pandas() requires an object exposing .to_numpy() or "
                f".values; got {type(obj)!r}"
            )

        values = np.asarray(values)

        if values.ndim == 2:

            if values.shape[1] != 1:

                raise ValueError(
                    "from_pandas() requires a Series or a single-column "
                    f"DataFrame; got shape {values.shape}"
                )

            values = values[:, 0]

        index = getattr(obj, "index", None)
        if index is not None:
            index = np.asarray(index)

        name = getattr(obj, "name", None)
        if name is None:
            columns = getattr(obj, "columns", None)

            if columns is not None and len(columns) == 1:
                name = columns[0]

        return cls(values, index=index, name=name, freq=freq)

    # -- Dunders ----------------------------------------------------------

    def __len__(self) -> int:
        """ Number of observations. """
        return int(self.values.size)

    def __getitem__(self, key: int | slice) -> float | PriceSeries:
        """ Index by position (int -> scalar, slice -> sub-series). """
        if isinstance(key, slice):

            return PriceSeries(
                self.values[key],
                index=self.index[key],
                name=self.name,
                freq=self.freq,
            )

        return float(self.values[key])

    def __array__(self, dtype: Any = None) -> NDArray:
        """ numpy interop: ``np.asarray(ps)`` returns the values. """
        if dtype is None:

            return self.values

        return self.values.astype(dtype)

    def __eq__(self, other: object) -> bool:
        """ Equality by values and index. """
        if not isinstance(other, PriceSeries):

            return NotImplemented

        return (
            np.array_equal(self.values, other.values, equal_nan=True)
            and np.array_equal(self.index, other.index)
        )

    def __repr__(self) -> str:
        """ Short representation with head/tail. """
        n = len(self)
        name = f" name={self.name!r}" if self.name else ""
        freq = f" freq={self.freq!r}" if self.freq else ""

        if n <= 6:
            body = ", ".join(f"{v:g}" for v in self.values)

        else:
            head = ", ".join(f"{v:g}" for v in self.values[:3])
            tail = ", ".join(f"{v:g}" for v in self.values[-3:])
            body = f"{head}, ..., {tail}"

        return f"PriceSeries([{body}], len={n}{name}{freq})"

    __hash__ = None  # type: ignore[assignment]

    # -- Helpers ----------------------------------------------------------

    def _new(self, values: NDArray, index: NDArray | None = None) -> PriceSeries:
        """ Build a derived series carrying name/freq. """
        return PriceSeries(
            values,
            index=self.index if index is None else index,
            name=self.name,
            freq=self.freq,
        )

    def copy(self) -> PriceSeries:
        """ Return a copy of this series. """
        return self._new(self.values.copy())

    # -- Finance core methods (price <-> return identities) ----------------

    def to_returns(
        self,
        kind: str = "pct",
        dropna: bool = True,
    ) -> PriceSeries:
        """ Convert a price path to a return series.

        Parameters
        ----------
        kind : {"pct", "log", "raw"}
            ``pct`` -> ``p_t / p_{t-1} - 1``; ``log`` -> ``ln(p_t / p_{t-1})``;
            ``raw`` -> ``p_t - p_{t-1}``.
        dropna : bool
            Drop the (undefined) first observation. If ``False``, a leading
            ``NaN`` is kept so the length is preserved.

        Returns
        -------
        PriceSeries
            The return series (strictly causal: ``r_t`` uses only ``p_t`` and
            ``p_{t-1}``).

        Raises
        ------
        ValueError
            For ``kind="pct"`` or ``"log"`` when any price is non-positive
            (``<= 0``). Both conventions divide by the previous price (and
            ``log`` takes its logarithm), so a zero or negative price would
            otherwise yield silent ``inf``/``nan`` with only a bare
            ``RuntimeWarning``. Use ``kind="raw"`` for series that legitimately
            cross or touch zero.

        Examples
        --------
        >>> PriceSeries([100., 110., 99.]).to_returns("pct").values
        array([ 0.1, -0.1])

        """
        p = self.values

        if kind == "pct":
            if np.any(p <= 0.0):

                raise ValueError(
                    "to_returns(kind='pct') requires strictly positive prices; "
                    "a zero or negative price yields inf/nan. Use kind='raw' "
                    "for series that touch or cross zero."
                )

            r = p[1:] / p[:-1] - 1.0

        elif kind == "log":
            if np.any(p <= 0.0):

                raise ValueError(
                    "to_returns(kind='log') requires strictly positive prices; "
                    "log of a non-positive ratio is undefined. Use kind='raw' "
                    "for series that touch or cross zero."
                )

            r = np.log(p[1:] / p[:-1])

        elif kind == "raw":
            r = p[1:] - p[:-1]

        else:

            raise ValueError(f"unknown kind: {kind!r}")

        if dropna:

            return self._new(r, index=self.index[1:])

        if p.size == 0:
            # Empty input: no leading NaN to write (full[0] would be OOB).

            return self._new(np.empty(0, dtype=np.float64))

        full = np.empty(p.size, dtype=np.float64)
        full[0] = np.nan
        full[1:] = r

        return self._new(full)

    def to_prices(self, base: float = 1.0, kind: str = "pct") -> PriceSeries:
        """ Reconstruct a price path from this return series (inverse of
        :meth:`to_returns`).

        Parameters
        ----------
        base : float
            Initial price prepended to the path.
        kind : {"pct", "log", "raw"}
            Must match the convention the returns were computed with.

        Returns
        -------
        PriceSeries
            Price path of length ``len(self) + 1``. The index is **reset** to
            the default ``0..n`` range: the reconstructed path has one extra
            leading point (the ``base`` price) with no corresponding timestamp
            in the original return index, so a meaningful index cannot be
            carried through. Re-attach an index explicitly if needed.

        """
        r = self.values

        if kind == "pct":
            path = base * np.cumprod(1.0 + r)

        elif kind == "log":
            path = base * np.exp(np.cumsum(r))

        elif kind == "raw":
            path = base + np.cumsum(r)

        else:

            raise ValueError(f"unknown kind: {kind!r}")

        prices = np.empty(r.size + 1, dtype=np.float64)
        prices[0] = base
        prices[1:] = path

        return PriceSeries(prices, name=self.name, freq=self.freq)

    def cumulative(self) -> PriceSeries:
        """ Equity curve of this return series: ``cumprod(1 + r)``. """
        return self._new(np.cumprod(1.0 + self.values))

    def pnl(self, positions: ArrayLike) -> PriceSeries:
        """ Strategy returns of a position book on this return series.

        Causal: the position is shifted forward one step, so the position
        decided at ``t-1`` earns the return at ``t``
        (``pnl_t = position_{t-1} * r_t``). The first observation is ``NaN``
        (no prior position).

        Parameters
        ----------
        positions : array-like
            Position series aligned with this return series.

        Returns
        -------
        PriceSeries
            Strategy return series.

        """
        pos = np.asarray(positions, dtype=np.float64).reshape(-1)

        if pos.size != self.values.size:

            raise ValueError(
                f"positions length {pos.size} != returns length "
                f"{self.values.size}"
            )

        if pos.size == 0:
            # Empty input: no leading NaN to write (shifted[0] would be OOB).

            return self._new(np.empty(0, dtype=np.float64))

        shifted = np.empty_like(pos)
        shifted[0] = np.nan
        shifted[1:] = pos[:-1]

        return self._new(shifted * self.values)

    def drop_na(self) -> PriceSeries:
        """ Drop ``NaN`` observations (and their index entries). """
        mask = ~np.isnan(self.values)

        return self._new(self.values[mask], index=self.index[mask])

    def fillna(self, value: float = 0.0) -> PriceSeries:
        """ Replace ``NaN`` observations by a constant value. """
        return self._new(np.nan_to_num(self.values, nan=value))

    # -- Bridges & composition --------------------------------------------

    def to_numpy(self, copy: bool = True) -> NDArray[np.float64]:
        """ Return the values as a numpy array (copied by default). """
        return self.values.copy() if copy else self.values

    def to_torch(self, dtype: Any = None, device: Any = None) -> Any:
        """ Return the values as a ``torch.Tensor`` (lazy torch import). """
        import torch

        if dtype is None:
            dtype = torch.float32

        # Copy: the stored array is read-only, which torch warns about.
        return torch.as_tensor(self.values.copy(), dtype=dtype, device=device)

    def to_pandas(self) -> Any:
        """ Return this series as a :class:`pandas.Series` (lazy import).

        Raises
        ------
        ImportError
            If pandas is not installed, with a clear, actionable message.

        Examples
        --------
        >>> PriceSeries([100., 101., 99.], name="px").to_pandas()
        0    100.0
        1    101.0
        2     99.0
        Name: px, dtype: float64

        """
        try:
            import pandas as pd

        except ImportError as exc:

            raise ImportError("install pandas to use to_pandas()") from exc

        return pd.Series(self.values, index=self.index, name=self.name)

    def to_polars(self) -> Any:
        """ Return this series as a :class:`polars.Series` (lazy import).

        Raises
        ------
        ImportError
            If polars is not installed, with a clear, actionable message.
            (polars is a hard dependency of ``fynance``, so this branch is not
            normally reachable in this package's own environment; it exists so
            the method degrades gracefully wherever it is vendored or reused.)

        Examples
        --------
        >>> PriceSeries([1., 2., 3.], name="px").to_polars().to_list()
        [1.0, 2.0, 3.0]

        """
        try:
            import polars as pl

        except ImportError as exc:

            raise ImportError("install polars to use to_polars()") from exc

        return pl.Series(self.name, self.values)

    def pipe(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """ Apply a free function to the values, re-wrapping when possible.

        Delegates to the rich transform library without bloating this class:
        ``ps.pipe(rsi, window=14)`` calls ``rsi(ps.values, window=14)``. If
        the result is a 1-D array of the same length it is wrapped back into a
        :class:`PriceSeries` (same index); otherwise the raw result is
        returned.
        """
        out = func(self.values, *args, **kwargs)
        arr = np.asarray(out)

        if arr.ndim == 1 and arr.size == self.values.size:

            return self._new(arr)

        return out

    def apply(self, func: Callable[[float], float]) -> PriceSeries:
        """ Apply an element-wise function to the values.

        On an empty series this short-circuits to an empty result (``np.vectorize``
        cannot infer an output dtype from zero inputs).
        """
        if self.values.size == 0:

            return self._new(self.values.copy())

        return self._new(np.vectorize(func)(self.values))
