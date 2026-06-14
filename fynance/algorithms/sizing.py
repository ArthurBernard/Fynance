#!/usr/bin/env python3
# coding: utf-8

""" Position sizing and transaction-cost primitives for realistic backtests.

Time-series leverage rules (Kelly, volatility targeting) and a turnover-based
transaction-cost model — the building blocks for going from a raw signal to a
net-of-cost P&L.

Main entry points
-----------------
- :func:`kelly_fraction` — (fractional) Kelly leverage from return moments.
- :func:`vol_target` — causal leverage that targets a constant volatility.
- :func:`transaction_cost` — per-step cost from weight turnover.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.features.indicators import realized_volatility

__all__ = ['kelly_fraction', 'transaction_cost', 'vol_target']


def kelly_fraction(returns: NDArray, fraction: float = 1.0) -> float:
    r""" Fractional Kelly leverage from a return series.

    Under a Gaussian approximation the growth-optimal leverage is
    :math:`f^\star = \mu / \sigma^2`; ``fraction`` scales it down
    (fractional Kelly, e.g. 0.5 for half-Kelly).

    Parameters
    ----------
    returns : array_like
        Series of (arithmetic) returns.
    fraction : float, optional
        Multiplier on the full Kelly leverage. Default 1.0.

    Returns
    -------
    float
        Kelly leverage (0 if the variance is null).

    Examples
    --------
    >>> import numpy as np
    >>> r = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
    >>> round(kelly_fraction(r, fraction=0.5), 4)
    13.5135

    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    var = returns.var(ddof=0)
    if var <= 0:
        return 0.0

    return float(fraction * returns.mean() / var)


def vol_target(
    X: NDArray,
    target_vol: float = 0.15,
    period: int = 252,
    w: int = 21,
    max_leverage: float = 5.0,
) -> NDArray:
    r""" Causal volatility-targeting leverage series.

    Leverage that scales inversely with the *past* realized volatility so the
    strategy targets a constant annualized volatility ``target_vol``:
    :math:`\\ell_t = target\\_vol / \\hat\\sigma_t`, capped at ``max_leverage``.
    Strictly causal — uses :func:`fynance.features.indicators.realized_volatility`.

    Parameters
    ----------
    X : array_like
        Price/level series.
    target_vol : float, optional
        Target annualized volatility. Default 0.15.
    period : int, optional
        Annualization factor. Default 252.
    w : int, optional
        Rolling window for the realized volatility. Default 21.
    max_leverage : float, optional
        Cap on the leverage. Default 5.0.

    Returns
    -------
    np.ndarray
        Leverage series aligned to ``X`` (0 where volatility is not yet defined).

    """
    X = np.asarray(X, dtype=np.float64)
    vol = np.asarray(realized_volatility(X, w=w, period=period))
    with np.errstate(divide='ignore', invalid='ignore'):
        lev = np.where(vol > 0, target_vol / vol, 0.0)

    return np.clip(lev, 0.0, max_leverage)


def transaction_cost(
    weights: NDArray, fee: float = 0.001, axis: int = 0,
) -> NDArray:
    r""" Per-step transaction cost from weight turnover.

    Cost at each step is ``fee`` times the traded amount (turnover):
    :math:`c_t = fee \\cdot \\sum_i |w_{t,i} - w_{t-1,i}|`, with the first step
    charging the initial position.

    Parameters
    ----------
    weights : array_like
        Portfolio weights over time, shape ``(T,)`` or ``(T, n_assets)``.
    fee : float, optional
        Proportional cost per unit traded (e.g. 0.001 = 10 bps). Default 0.001.
    axis : {0, 1}, optional
        Time axis. Default 0.

    Returns
    -------
    np.ndarray
        Cost per step, shape ``(T,)``.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]])
    >>> transaction_cost(w, fee=0.01)
    array([0.01, 0.01, 0.  ])

    """
    w = np.asarray(weights, dtype=np.float64)
    if axis == 1:
        w = w.T
    if w.ndim == 1:
        w = w.reshape(-1, 1)

    turnover = np.empty(w.shape[0])
    turnover[0] = np.abs(w[0]).sum()
    turnover[1:] = np.abs(np.diff(w, axis=0)).sum(axis=1)

    return fee * turnover
