#!/usr/bin/env python3
# coding: utf-8

""" Core value objects and composition contracts.

.. currentmodule:: fynance.core

Exposes :class:`PriceSeries` (the central numpy-backed financial time-series),
:class:`OHLCV` (the aligned multi-series Open/High/Low/Close/Volume container)
and the :mod:`typing.Protocol` seams the pipeline composes through.

"""

# Local packages
from .ohlcv import OHLCV
from .price_series import PriceSeries
from .protocols import (
    Allocator,
    CostModel,
    DataSource,
    FeatureTransform,
    Metric,
    SignalModel,
)

__all__ = [
    'PriceSeries',
    'OHLCV',
    'DataSource',
    'FeatureTransform',
    'SignalModel',
    'Allocator',
    'CostModel',
    'Metric',
]
