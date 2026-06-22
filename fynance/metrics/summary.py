#!/usr/bin/env python3
# coding: utf-8

""" Aggregated performance summary and a metric registry.

A single :func:`summary` reduces an equity/price curve to the standard panel of
risk-adjusted metrics, and :data:`METRICS` exposes them by name.

"""

from __future__ import annotations

# Built-in packages
from typing import Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.metrics.drawdown import mdd
from fynance.metrics.ratios import (
    annual_volatility,
    calmar,
    sharpe,
    sortino,
)
from fynance.metrics.returns import annual_return

__all__ = ['METRICS', 'summary']

# Registry of scalar metrics taking an equity/price curve -> float.
METRICS: dict[str, Callable[..., NDArray]] = {
    'annual_return': annual_return,
    'annual_volatility': annual_volatility,
    'sharpe': sharpe,
    'sortino': sortino,
    'calmar': calmar,
    'max_drawdown': mdd,
}


def summary(prices: NDArray, period: int = 252, rf: float = 0.0) -> dict[str, float]:
    """ Standard performance summary of an equity/price curve.

    Parameters
    ----------
    prices : array-like
        Equity or price curve (not returns).
    period : int
        Annualization factor (252 for daily data).
    rf : float
        Risk-free rate passed to the Sharpe ratio.

    Returns
    -------
    dict of str to float
        ``annual_return``, ``annual_volatility``, ``sharpe``, ``sortino``,
        ``calmar`` and ``max_drawdown``.

    Examples
    --------
    >>> import numpy as np
    >>> eq = np.array([100., 101., 103., 102., 105., 107.])
    >>> s = summary(eq)
    >>> sorted(s)
    ['annual_return', 'annual_volatility', 'calmar', 'max_drawdown', 'sharpe', 'sortino']

    """
    p = np.asarray(prices, dtype=np.float64)

    # ``sharpe`` and ``sortino`` default to ``log=False``; report the volatility
    # under the same convention so the displayed vol is the one implied by the
    # displayed Sharpe ratio.
    log = False

    return {
        'annual_return': float(annual_return(p, period=period)),
        'annual_volatility': float(annual_volatility(p, period=period, log=log)),
        'sharpe': float(sharpe(p, period=period, rf=rf, log=log)),
        'sortino': float(sortino(p, period=period, log=log)),
        'calmar': float(calmar(p, period=period)),
        'max_drawdown': float(mdd(p)),
    }
