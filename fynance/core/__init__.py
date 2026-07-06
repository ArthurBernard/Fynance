#!/usr/bin/env python3
# coding: utf-8

""" Core value objects and composition contracts.

.. currentmodule:: fynance.core

Exposes :class:`PriceSeries` (the central numpy-backed financial time-series),
:class:`OHLCV` (the aligned multi-series Open/High/Low/Close/Volume container),
the :mod:`typing.Protocol` seams the pipeline composes through, and two
executable house-rule checks for them: :func:`check_conforms` (protocol
conformance smoke test) and :func:`assert_causal` (no-lookahead probe).

"""

# Local packages
from .checks import assert_causal, check_conforms
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
    'check_conforms',
    'assert_causal',
]
