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


def _annualized_sharpe(returns: NDArray, period: int = 252) -> float:
    """ Annualized Sharpe ratio of a return series (numpy, self-contained). """
    r = returns[~np.isnan(returns)]
    sd = r.std()

    if sd == 0:

        return 0.0

    return float(np.sqrt(period) * r.mean() / sd)


def _annualized_sortino(returns: NDArray, period: int = 252) -> float:
    """ Annualized Sortino ratio (downside deviation denominator). """
    r = returns[~np.isnan(returns)]
    downside = r[r < 0]
    dd = downside.std() if downside.size else 0.0

    if dd == 0:

        return 0.0

    return float(np.sqrt(period) * r.mean() / dd)


def _max_drawdown(equity: NDArray) -> float:
    """ Maximum drawdown of an equity curve (fraction, >= 0). """
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / peak

    return float(np.max(dd))


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

    """

    equity: NDArray
    returns: NDArray
    gross_returns: NDArray
    positions: NDArray
    costs: NDArray
    index: NDArray | None = None

    def to_numpy(self) -> NDArray:
        """ Return the equity curve as a numpy array. """
        return np.asarray(self.equity)

    def to_price_series(self) -> PriceSeries:
        """ Return the equity curve as a :class:`PriceSeries`. """
        return PriceSeries(self.equity, index=self.index, name="equity")

    def summary(self, period: int = 252) -> dict[str, float]:
        """ Standard performance summary.

        Returns a dict with annualized return/volatility, Sharpe, Sortino,
        max drawdown, Calmar, hit-rate and total transaction cost.
        """
        r = self.returns[~np.isnan(self.returns)]
        ann_ret = float(r.mean() * period)
        ann_vol = float(r.std() * np.sqrt(period))
        mdd = _max_drawdown(self.equity)

        return {
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe": _annualized_sharpe(self.returns, period),
            "sortino": _annualized_sortino(self.returns, period),
            "max_drawdown": mdd,
            "calmar": float(ann_ret / mdd) if mdd > 0 else 0.0,
            "hit_rate": float((r > 0).mean()) if r.size else 0.0,
            "total_cost": float(np.nansum(self.costs)),
        }
