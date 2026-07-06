#!/usr/bin/env python3
# coding: utf-8

""" Data ingestion layer: ports & adapters, alignment, temporal splits.

.. currentmodule:: fynance.data

The only I/O boundary of the library. File adapters turn local CSV/Parquet into
:class:`~fynance.core.PriceSeries`; :func:`align`/:func:`resample` reconcile
multiple series; :func:`train_test_split`/:func:`walk_forward`/
:func:`combinatorial_purged_cv` build no-lookahead evaluation indices.

"""

# Local packages
from .align import align, resample
from .base import BaseDataSource, get_source, load, register
from .csv import CSVSource
from .parquet import ParquetSource
from .split import combinatorial_purged_cv, train_test_split, walk_forward

__all__ = [
    'BaseDataSource',
    'register',
    'get_source',
    'load',
    'CSVSource',
    'ParquetSource',
    'align',
    'resample',
    'train_test_split',
    'walk_forward',
    'combinatorial_purged_cv',
]
