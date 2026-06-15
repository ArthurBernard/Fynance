#!/usr/bin/env python3
# coding: utf-8

""" Shared helpers for the plotting layer. """

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = ['as_equity']


def as_equity(obj: object) -> tuple[NDArray, NDArray | None]:
    """ Coerce a result/series/array into an ``(equity, index)`` pair.

    Accepts a :class:`~fynance.backtest.result.BacktestResult`, a
    :class:`~fynance.core.PriceSeries`, or a raw array (treated as an equity
    curve).
    """
    equity = getattr(obj, "equity", None)

    if equity is not None:  # BacktestResult
        index = getattr(obj, "index", None)

        return np.asarray(equity, dtype=np.float64), index

    values = getattr(obj, "values", None)

    if values is not None:  # PriceSeries
        return np.asarray(values, dtype=np.float64), getattr(obj, "index", None)

    return np.asarray(obj, dtype=np.float64), None


def drawdown_curve(equity: NDArray) -> NDArray:
    """ Underwater drawdown curve (fraction below running peak, <= 0). """
    peak = np.maximum.accumulate(equity)

    return equity / peak - 1.0
