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

    """
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

    Parameters
    ----------
    ps : PriceSeries
        Series with a datetime index.
    freq : str
        Target polars frequency (e.g. ``"1w"``, ``"1mo"``).
    agg : {"last", "mean", "ohlc"}
        Aggregation. ``ohlc`` returns a mapping with open/high/low/close.

    """
    import polars as pl

    df = pl.DataFrame({"_t": ps.index, "_v": ps.values}).sort("_t")
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
