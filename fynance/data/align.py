#!/usr/bin/env python3
# coding: utf-8

""" Multi-asset alignment and frequency resampling.

All operations are causal: forward-fill uses only past values, and downsampling
aggregations never look past a period's right edge.

"""

from __future__ import annotations

# Built-in packages
# Third-party packages
import numpy as np

# Local packages
from fynance.core import PriceSeries

__all__ = ['align', 'resample']


def _ffill(values: np.ndarray) -> np.ndarray:
    """ Forward-fill NaNs using past values only. """
    out = values.copy()
    mask = np.isnan(out)
    idx = np.where(~mask, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    # Leading NaNs (before the first valid value) stay NaN.
    first_valid = np.argmax(~mask) if mask.any() and (~mask).any() else 0
    out[:first_valid] = np.nan

    return out


def align(
    series: dict[str, PriceSeries],
    how: str = "outer",
    fill: str | None = "ffill",
) -> dict[str, PriceSeries]:
    """ Align several series onto a common index.

    Parameters
    ----------
    series : dict of str to PriceSeries
        Named series to align.
    how : {"outer", "inner"}
        ``outer`` uses the union of indices, ``inner`` the intersection.
    fill : {"ffill", None}
        Forward-fill (past-only) missing values after an outer align.

    Returns
    -------
    dict of str to PriceSeries
        Series sharing a common index.

    Raises
    ------
    ValueError
        If ``how`` is unknown, or if any input series has duplicate index
        entries. Duplicate timestamps would otherwise be silently collapsed
        (the index-to-value mapping keeps only the last value), shrinking the
        series to its unique count without warning.

    """
    for name, ps in series.items():
        idx_list = ps.index.tolist()

        if len(idx_list) != len(set(idx_list)):

            raise ValueError(
                f"series {name!r} has duplicate index entries; align cannot "
                "map a timestamp to more than one value (deduplicate or "
                "aggregate the series first)"
            )

    index_sets = [set(ps.index.tolist()) for ps in series.values()]

    if how == "outer":
        common = sorted(set().union(*index_sets))

    elif how == "inner":
        common = sorted(set(index_sets[0]).intersection(*index_sets[1:]))

    else:

        raise ValueError(f"unknown how: {how!r}")

    common_arr = np.array(common)
    out: dict[str, PriceSeries] = {}

    for name, ps in series.items():
        mapping = dict(zip(ps.index.tolist(), ps.values.tolist()))
        vals = np.array([mapping.get(t, np.nan) for t in common], dtype=float)

        if fill == "ffill" and how == "outer":
            vals = _ffill(vals)

        out[name] = PriceSeries(vals, index=common_arr, name=name, freq=ps.freq)

    return out


def resample(
    ps: PriceSeries,
    freq: str,
    agg: str = "last",
) -> PriceSeries | dict[str, PriceSeries]:
    """ Downsample a series to a coarser frequency.

    The series index must be a NumPy ``datetime64`` array; polars'
    ``group_by_dynamic`` resampling is defined only over a temporal axis. An
    integer or object-dtype (e.g. ``datetime.datetime``) index is rejected with
    an explanatory :class:`ValueError` rather than surfacing an opaque polars
    error.

    polars only accepts ``datetime64`` resolutions ``[D]``, ``[ms]``, ``[us]``
    and ``[ns]``. Any other resolution (the common ``[s]``, plus ``[h]``,
    ``[m]``, ``[W]``, ``[M]``, ``[Y]``) is losslessly upcast to ``[us]`` before
    resampling, so it works instead of triggering an opaque polars error.

    Parameters
    ----------
    ps : PriceSeries
        Series with a ``datetime64`` index.
    freq : str
        Target polars frequency (e.g. ``"1w"``, ``"1mo"``).
    agg : {"last", "mean", "ohlc"}
        Aggregation. ``ohlc`` returns a mapping with open/high/low/close.

    Returns
    -------
    PriceSeries or dict of str to PriceSeries
        The resampled series (or an open/high/low/close mapping for ``ohlc``).

    Raises
    ------
    ValueError
        If the index is not a ``datetime64`` array, or if ``agg`` is unknown.

    """
    import polars as pl

    index = np.asarray(ps.index)

    if index.dtype.kind != "M":

        raise ValueError(
            "resample requires a datetime64 index, got an index of dtype "
            f"{index.dtype!r}; convert the index to numpy datetime64 first "
            "(integer and object/datetime.datetime indexes are not supported)"
        )

    # polars only accepts datetime64 [D]/[ms]/[us]/[ns]; upcast any other
    # resolution (e.g. the common [s], or [h]/[m]/[W]/[M]/[Y]) to [us] so the
    # call succeeds instead of raising an opaque polars resolution error.
    _supported = (
        np.dtype("datetime64[D]"),
        np.dtype("datetime64[ms]"),
        np.dtype("datetime64[us]"),
        np.dtype("datetime64[ns]"),
    )

    if index.dtype not in _supported:
        index = index.astype("datetime64[us]")

    df = pl.DataFrame({"_t": index, "_v": ps.values}).sort("_t")
    gb = df.group_by_dynamic("_t", every=freq)

    if agg == "last":
        res = gb.agg(pl.col("_v").last())

        return PriceSeries(res["_v"].to_numpy(), index=res["_t"].to_numpy(),
                           name=ps.name, freq=freq)

    if agg == "mean":
        res = gb.agg(pl.col("_v").mean())

        return PriceSeries(res["_v"].to_numpy(), index=res["_t"].to_numpy(),
                           name=ps.name, freq=freq)

    if agg == "ohlc":
        res = gb.agg(
            pl.col("_v").first().alias("open"),
            pl.col("_v").max().alias("high"),
            pl.col("_v").min().alias("low"),
            pl.col("_v").last().alias("close"),
        )
        idx = res["_t"].to_numpy()

        return {
            c: PriceSeries(res[c].to_numpy(), index=idx, name=c, freq=freq)
            for c in ("open", "high", "low", "close")
        }

    raise ValueError(f"unknown agg: {agg!r}")
