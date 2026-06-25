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
    cost_components : dict of str to numpy.ndarray, optional
        Per-step cost broken down by component (e.g. ``transaction`` vs
        ``market_impact``); the values sum to :attr:`costs`. Populated when the
        cost model exposes the optional ``components`` convention, else ``None``.

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
    cost_components: dict[str, NDArray] | None = None

    def to_numpy(self) -> NDArray:
        """ Return the equity curve as a numpy array. """
        return np.asarray(self.equity)

    def to_price_series(self) -> PriceSeries:
        """ Return the equity curve as a :class:`PriceSeries`. """
        return PriceSeries(self.equity, index=self.index, name="equity")

    def summary(self, period: int = 252) -> dict[str, float]:
        """ Standard performance summary.

        Delegates the risk-adjusted ratios and drawdown to
        :func:`fynance.metrics.summary` (computed on the equity curve) and adds,
        from the strategy's own data, the hit-rate, total transaction cost and
        the trading-profile churn (``n_sign_changes`` / ``trades_per_year``,
        summed over the book — see :func:`fynance.metrics.sign_changes`).
        """
        from fynance.metrics import summary as _metric_summary
        from fynance.metrics.trading import sign_changes, trades_per_year

        out = _metric_summary(self.equity, period=period)
        r = self.returns[~np.isnan(self.returns)]
        out["hit_rate"] = float((r > 0).mean()) if r.size else 0.0
        out["total_cost"] = float(np.nansum(self.costs))
        out["n_sign_changes"] = float(np.sum(sign_changes(self.positions)))
        out["trades_per_year"] = float(
            np.sum(trades_per_year(self.positions, period=period)))

        return out
