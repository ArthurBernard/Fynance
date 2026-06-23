#!/usr/bin/env python3
# coding: utf-8

""" Backtest result value object.

:class:`BacktestResult` is the engine's output and the hand-off to metrics and
reporting. It holds numpy arrays and computes a standard performance summary.

"""

from __future__ import annotations

# Built-in packages
from dataclasses import dataclass

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.core import PriceSeries

__all__ = ['BacktestResult']


@dataclass
class BacktestResult:
    """ Output of :func:`~fynance.backtest.engine.backtest`.

    Attributes
    ----------
    equity : numpy.ndarray
        Equity curve.
    returns : numpy.ndarray
        Net strategy returns (after costs).
    gross_returns : numpy.ndarray
        Strategy returns before costs.
    positions : numpy.ndarray
        Position/weight book used.
    costs : numpy.ndarray
        Per-step transaction costs.
    index : numpy.ndarray, optional
        Temporal index carried from the input.
    asset_gross_returns : numpy.ndarray, optional
        Per-asset gross return contributions ``(T, N)`` for a multi-asset book
        (they sum to :attr:`gross_returns`); ``None`` for a single-asset run.

    Methods
    -------
    to_numpy
    to_price_series
    summary

    """

    equity: NDArray
    returns: NDArray
    gross_returns: NDArray
    positions: NDArray
    costs: NDArray
    index: NDArray | None = None
    asset_gross_returns: NDArray | None = None

    def to_numpy(self) -> NDArray:
        """ Return the equity curve as a numpy array. """
        return np.asarray(self.equity)

    def to_price_series(self) -> PriceSeries:
        """ Return the equity curve as a :class:`PriceSeries`. """
        return PriceSeries(self.equity, index=self.index, name="equity")

    def summary(self, period: int = 252) -> dict[str, float]:
        """ Standard performance summary.

        Delegates the risk-adjusted ratios and drawdown to
        :func:`fynance.metrics.summary` (computed on the equity curve) and adds
        the hit-rate and total transaction cost from the strategy's own data.
        """
        from fynance.metrics import summary as _metric_summary

        out = _metric_summary(self.equity, period=period)
        r = self.returns[~np.isnan(self.returns)]
        out["hit_rate"] = float((r > 0).mean()) if r.size else 0.0
        out["total_cost"] = float(np.nansum(self.costs))

        return out
