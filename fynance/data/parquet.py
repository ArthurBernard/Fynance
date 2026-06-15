#!/usr/bin/env python3
# coding: utf-8

""" Parquet data-source adapter (polars-backed). """

from __future__ import annotations

# Built-in packages
from pathlib import Path
from typing import Any

# Local packages
from fynance.core import PriceSeries
from fynance.data.base import BaseDataSource, frame_to_series, register

__all__ = ['ParquetSource']


@register("parquet")
class ParquetSource(BaseDataSource):
    """ Read a Parquet file into a :class:`PriceSeries` (or a mapping). """

    def load(
        self,
        path: str | Path,
        value_col: str | None = None,
        index_col: str | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> PriceSeries | dict[str, PriceSeries]:
        """ Load ``path`` (see :meth:`CSVSource.load` for parameters). """
        import polars as pl

        df = pl.read_parquet(path, **kwargs)

        return frame_to_series(df, value_col, index_col, freq)
