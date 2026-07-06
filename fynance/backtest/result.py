#!/usr/bin/env python3
# coding: utf-8

""" Backtest result value object.

:class:`BacktestResult` is the engine's output and the hand-off to metrics and
reporting. It holds numpy arrays and computes a standard performance summary.

"""

from __future__ import annotations

# Built-in packages
from dataclasses import dataclass
from typing import Any

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
    to_pandas
    summary
    trades
    trade_summary

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

    def to_pandas(self) -> Any:
        """ Return this result as a :class:`pandas.DataFrame` (lazy import).

        One row per time step. Always carries ``equity``, ``returns``,
        ``gross_returns`` and ``costs``. :attr:`positions` becomes a single
        ``positions`` column for a single-asset ``(T,)`` book, or
        ``pos_0``..``pos_{N-1}`` columns for a multi-asset ``(T, N)`` book.
        When present, :attr:`asset_gross_returns` similarly expands into
        ``asset_gross_return_0``..``asset_gross_return_{N-1}`` columns, and
        each :attr:`cost_components` entry becomes a ``cost_<name>`` column.
        The DataFrame index is :attr:`index` when set, else pandas' default
        ``RangeIndex``.

        Raises
        ------
        ImportError
            If pandas is not installed, with a clear, actionable message.

        Examples
        --------
        >>> from fynance.backtest import backtest
        >>> res = backtest(np.array([0.01, 0.02, -0.01]), np.ones(3))
        >>> list(res.to_pandas().columns)
        ['equity', 'returns', 'gross_returns', 'costs', 'positions']

        """
        try:
            import pandas as pd

        except ImportError as exc:

            raise ImportError("install pandas to use to_pandas()") from exc

        data: dict[str, NDArray] = {
            "equity": self.equity,
            "returns": self.returns,
            "gross_returns": self.gross_returns,
            "costs": self.costs,
        }

        if self.positions.ndim == 2:
            for i in range(self.positions.shape[1]):
                data[f"pos_{i}"] = self.positions[:, i]

        else:
            data["positions"] = self.positions

        if self.asset_gross_returns is not None:
            for i in range(self.asset_gross_returns.shape[1]):
                data[f"asset_gross_return_{i}"] = self.asset_gross_returns[:, i]

        if self.cost_components is not None:
            for name, values in self.cost_components.items():
                data[f"cost_{name}"] = values

        return pd.DataFrame(data, index=self.index)

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

    def trades(self) -> NDArray:
        """ Round-trip trades extracted from the strategy's own arrays.

        Delegates to :func:`fynance.metrics.trades.extract_trades` on
        :attr:`positions` and :attr:`returns`, taken exactly as stored (see
        that function's docstring for the alignment convention this relies
        on).

        Returns
        -------
        numpy.ndarray
            Structured array, one row per trade -- see
            :func:`~fynance.metrics.trades.extract_trades`.
        """
        from fynance.metrics.trades import extract_trades

        return extract_trades(self.positions, self.returns)

    def trade_summary(self) -> dict[str, float]:
        """ Summary statistics of the strategy's round-trip trades.

        Delegates to :func:`fynance.metrics.trades.trade_summary` on
        :meth:`trades`.

        Returns
        -------
        dict of str to float
            See :func:`~fynance.metrics.trades.trade_summary`.
        """
        from fynance.metrics.trades import trade_summary as _trade_summary

        return _trade_summary(self.trades())
