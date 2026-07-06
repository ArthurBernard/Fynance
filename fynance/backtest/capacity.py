#!/usr/bin/env python3
# coding: utf-8

""" Capacity analysis: net Sharpe vs AUM, and the breakeven fee.

Two questions a strategy owner asks before sizing up: how does performance
degrade as more capital is deployed (:func:`capacity_curve`), and how large a
proportional fee can the strategy absorb before it stops being worth trading
(:func:`breakeven_fee`)? Both reuse :func:`~fynance.backtest.engine.backtest`
unchanged — capacity is a sweep over cost models, not a new engine.

"""

from __future__ import annotations

# Built-in packages
from typing import Any, Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.backtest.cost import ProportionalCost
from fynance.backtest.engine import backtest

__all__ = ['capacity_curve', 'breakeven_fee']

#: Fee cap (proportional, per unit traded) above which :func:`breakeven_fee`
#: gives up and reports the strategy has no breakeven fee below it.
_FEE_CAP = 0.10


def capacity_curve(
    weights: Any,
    X: Any,
    aums: Any,
    cost_factory: Callable[[float], Any],
    period: int = 252,
) -> dict[str, NDArray[np.float64]]:
    """ Sweep net performance across a book of AUM levels.

    Runs :func:`~fynance.backtest.engine.backtest` once per AUM level, each
    time with the cost model ``cost_factory`` builds for that level. The
    caller encodes how cost should scale with size (e.g. a market-impact
    coefficient growing with the square root of AUM) inside ``cost_factory``;
    this function only orchestrates the sweep and collects the results.

    Parameters
    ----------
    weights : array-like
        Position/weight book, shape ``(T,)`` for a single asset or
        ``(T, N)`` for a multi-asset book.
    X : array-like
        Price levels aligned with ``weights`` (same shape); passed to
        :func:`~fynance.backtest.engine.backtest` as ``returns_input=False``.
    aums : array-like
        1-D array of AUM levels to sweep, typically increasing.
    cost_factory : callable
        Maps an AUM level (float) to a cost model (e.g.
        :class:`~fynance.backtest.cost.MarketImpactCost`) conforming to
        :class:`~fynance.core.protocols.CostModel`.
    period : int
        Annualization factor forwarded to
        :meth:`~fynance.backtest.result.BacktestResult.summary`.

    Returns
    -------
    dict of str to numpy.ndarray
        ``aum``, ``net_sharpe``, ``net_annual_return`` and ``total_cost``,
        one entry per AUM level, aligned with ``aums``.

    Raises
    ------
    ValueError
        If ``aums`` is not 1-D, or ``weights`` and ``X`` do not share the
        same shape.

    Examples
    --------
    A single-asset strategy with a small planted edge (``0.06%`` of the
    signal per bar) and turnover from a noisy signal, swept across AUM
    levels with a square-root market-impact factory: net Sharpe degrades
    monotonically as impact grows with size.

    >>> import numpy as np
    >>> from fynance.backtest.cost import MarketImpactCost
    >>> rng = np.random.default_rng(0)
    >>> steps = np.cumsum(rng.normal(0.0, 0.07, size=300))
    >>> weights = np.clip(steps, -1.0, 1.0)
    >>> noise = rng.normal(0.0, 0.01, size=300)
    >>> returns = np.empty(300)
    >>> returns[0] = noise[0]
    >>> returns[1:] = 0.0006 * weights[:-1] + noise[1:]
    >>> prices = 100.0 * np.cumprod(1.0 + returns)
    >>> aums = np.array([1e5, 1e7, 1e9])
    >>> cost_factory = lambda aum: MarketImpactCost(
    ...     impact=0.0017 * np.sqrt(aum / 1e6), exponent=1.5
    ... )
    >>> out = capacity_curve(weights, prices, aums, cost_factory)
    >>> sorted(out)
    ['aum', 'net_annual_return', 'net_sharpe', 'total_cost']
    >>> bool(np.all(np.diff(out['net_sharpe']) <= 1e-9))
    True

    """
    w = np.asarray(weights, dtype=np.float64)
    prices = np.asarray(X, dtype=np.float64)
    aum_arr = np.asarray(aums, dtype=np.float64)

    if aum_arr.ndim != 1:

        raise ValueError(f"aums must be 1-D, got shape {aum_arr.shape}")

    if w.shape != prices.shape:

        raise ValueError(
            f"weights shape {w.shape} != X shape {prices.shape}"
        )

    n = aum_arr.shape[0]
    net_sharpe = np.empty(n, dtype=np.float64)
    net_annual_return = np.empty(n, dtype=np.float64)
    total_cost = np.empty(n, dtype=np.float64)

    for i, aum in enumerate(aum_arr):
        cost = cost_factory(float(aum))
        res = backtest(prices, w, cost=cost, returns_input=False)
        stats = res.summary(period=period)
        net_sharpe[i] = stats["sharpe"]
        net_annual_return[i] = stats["annual_return"]
        total_cost[i] = stats["total_cost"]

    return {
        "aum": aum_arr,
        "net_sharpe": net_sharpe,
        "net_annual_return": net_annual_return,
        "total_cost": total_cost,
    }


def breakeven_fee(
    weights: Any,
    X: Any,
    period: int = 252,
    tol: float = 1e-6,
) -> float:
    """ Proportional fee at which the net Sharpe ratio crosses zero.

    Bisects on a :class:`~fynance.backtest.cost.ProportionalCost` fee: the
    bracket's upper bound is doubled from a small seed until the net Sharpe
    turns non-positive, capped at :data:`_FEE_CAP` (10% per trade traded).

    Parameters
    ----------
    weights : array-like
        Position/weight book, shape ``(T,)`` or ``(T, N)``.
    X : array-like
        Price levels aligned with ``weights``; passed to
        :func:`~fynance.backtest.engine.backtest` as ``returns_input=False``.
    period : int
        Annualization factor for the Sharpe ratio.
    tol : float
        Bisection tolerance on the fee (absolute).

    Returns
    -------
    float
        The proportional fee at which the net Sharpe ratio is (approximately)
        zero.

    Raises
    ------
    ValueError
        If the gross (fee=0) Sharpe ratio is already non-positive (message
        ``'strategy unprofitable gross'``), or if the net Sharpe ratio is
        still positive at the 10%-per-trade cap (message ``'no breakeven
        below 10%'``).

    Examples
    --------
    Same planted-edge, noisy-turnover strategy as in :func:`capacity_curve`:
    a proportional fee somewhere below the 10%-per-trade cap wipes out the
    net Sharpe ratio.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> steps = np.cumsum(rng.normal(0.0, 0.07, size=1000))
    >>> weights = np.clip(steps, -1.0, 1.0)
    >>> noise = rng.normal(0.0, 0.01, size=1000)
    >>> returns = np.empty(1000)
    >>> returns[0] = noise[0]
    >>> returns[1:] = 0.0006 * weights[:-1] + noise[1:]
    >>> prices = 100.0 * np.cumprod(1.0 + returns)
    >>> fee = breakeven_fee(weights, prices)
    >>> bool(0.0 < fee < 0.10)
    True

    """

    def _sharpe(fee: float) -> float:
        cost = ProportionalCost(fee=fee) if fee > 0.0 else None
        res = backtest(X, weights, cost=cost, returns_input=False)

        return float(res.summary(period=period)["sharpe"])

    gross_sharpe = _sharpe(0.0)

    if gross_sharpe <= 0.0:

        raise ValueError("strategy unprofitable gross")

    lo, hi = 0.0, 1e-4

    while _sharpe(hi) > 0.0:

        if hi >= _FEE_CAP:

            raise ValueError("no breakeven below 10%")

        hi = min(2.0 * hi, _FEE_CAP)

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)

        if _sharpe(mid) > 0.0:
            lo = mid

        else:
            hi = mid

    return 0.5 * (lo + hi)
