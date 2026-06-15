#!/usr/bin/env python3
# coding: utf-8

""" Data-source port: base class, adapter registry and the ``load`` dispatcher.

:class:`DataSource` (in :mod:`fynance.core.protocols`) is the only I/O boundary
of the library. Concrete adapters (CSV, Parquet, ...) register here and the
:func:`load` dispatcher picks one by file extension.

"""

from __future__ import annotations

# Built-in packages
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

# Local packages
from fynance.core import PriceSeries

__all__ = ['BaseDataSource', 'register', 'get_source', 'load']

_REGISTRY: dict[str, type["BaseDataSource"]] = {}

# Map file extensions to registered source names.
_EXT = {
    ".csv": "csv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}


class BaseDataSource(ABC):
    """ Abstract base for data-source adapters (the I/O port). """

    @abstractmethod
    def load(self, path: str | Path, **kwargs: Any) -> Any:
        """ Load ``path`` into a :class:`PriceSeries` or a mapping of them. """
        ...


def register(name: str) -> Callable[[type[BaseDataSource]], type[BaseDataSource]]:
    """ Class decorator registering a data-source adapter under ``name``. """
    def deco(cls: type[BaseDataSource]) -> type[BaseDataSource]:
        _REGISTRY[name] = cls

        return cls

    return deco


def get_source(name: str) -> BaseDataSource:
    """ Instantiate a registered data source by name. """
    if name not in _REGISTRY:

        raise ValueError(
            f"unknown data source {name!r}; registered: {sorted(_REGISTRY)}"
        )

    return _REGISTRY[name]()


def load(
    path: str | Path,
    source: str = "auto",
    **kwargs: Any,
) -> PriceSeries | dict[str, PriceSeries]:
    """ Load a file into a :class:`PriceSeries` (or mapping for multi-column).

    Parameters
    ----------
    path : str or pathlib.Path
        File to read.
    source : str
        Adapter name, or ``"auto"`` to select by file extension.
    **kwargs
        Forwarded to the adapter's ``load``.

    """
    if source == "auto":
        ext = Path(path).suffix.lower()

        if ext not in _EXT:

            raise ValueError(
                f"cannot infer data source from extension {ext!r}; "
                f"pass source= explicitly"
            )

        source = _EXT[ext]

    return get_source(source).load(path, **kwargs)


def frame_to_series(
    df: Any,
    value_col: str | None = None,
    index_col: str | None = None,
    freq: str | None = None,
) -> PriceSeries | dict[str, PriceSeries]:
    """ Convert a polars DataFrame to a :class:`PriceSeries` or a mapping.

    Shared by the CSV and Parquet adapters: resolves the temporal index column
    and the value column(s).
    """
    import polars as pl

    if index_col is None:
        temporal = [
            c for c, dt in zip(df.columns, df.dtypes)
            if dt in (pl.Date, pl.Datetime)
        ]
        index_col = temporal[0] if temporal else None

    index = df[index_col].to_numpy() if index_col is not None else None
    value_cols = [c for c in df.columns if c != index_col]

    if value_col is not None:

        return PriceSeries(
            df[value_col].to_numpy(),
            index=index,
            name=value_col,
            freq=freq,
        )

    if len(value_cols) == 1:

        return PriceSeries(
            df[value_cols[0]].to_numpy(),
            index=index,
            name=value_cols[0],
            freq=freq,
        )

    return {
        c: PriceSeries(df[c].to_numpy(), index=index, name=c, freq=freq)
        for c in value_cols
    }
