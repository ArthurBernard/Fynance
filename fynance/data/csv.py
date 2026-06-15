#!/usr/bin/env python3
# coding: utf-8

""" CSV data-source adapter (polars-backed). """

from __future__ import annotations

# Built-in packages
from pathlib import Path
from typing import Any

# Local packages
from fynance.core import PriceSeries
from fynance.data.base import BaseDataSource, frame_to_series, register

__all__ = ['CSVSource']


@register("csv")
class CSVSource(BaseDataSource):
    """ Read a CSV file into a :class:`PriceSeries` (or a mapping). """

    def load(
        self,
        path: str | Path,
        value_col: str | None = None,
        index_col: str | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> PriceSeries | dict[str, PriceSeries]:
        """ Load ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            CSV file.
        value_col : str, optional
            Value column; if omitted, the single non-index column is used, or a
            mapping is returned for several value columns.
        index_col : str, optional
            Temporal index column; if omitted, the first date/datetime column.
        freq : str, optional
            Frequency tag carried onto the series.
        **kwargs
            Forwarded to :func:`polars.read_csv`.

        """
        import polars as pl

        df = pl.read_csv(path, try_parse_dates=True, **kwargs)

        return frame_to_series(df, value_col, index_col, freq)
