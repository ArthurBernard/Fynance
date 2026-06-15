#!/usr/bin/env python3
# coding: utf-8

""" Signal layer: prediction -> position.

.. currentmodule:: fynance.signal

The bridge between :mod:`fynance.models` and :mod:`fynance.backtest`. Mappers
turn model predictions into positions; :class:`SignalPipeline` composes a model
with a mapper.

"""

# Local packages
from .mappers import rank, sign, threshold, vol_target_position
from .pipeline import SignalPipeline

__all__ = [
    'sign',
    'threshold',
    'rank',
    'vol_target_position',
    'SignalPipeline',
]
