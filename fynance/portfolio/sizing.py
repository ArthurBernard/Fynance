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
- :func:`book_vol_target` — multi-asset counterpart, targets the book's vol.
- :func:`transaction_cost` — per-step cost from weight turnover.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.features.indicators import realized_volatility

__all__ = ['book_vol_target', 'kelly_fraction', 'transaction_cost', 'vol_target']


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
    :math:`\ell_t = target\_vol / \hat\sigma_t`, capped at ``max_leverage``.
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


def book_vol_target(
    W: NDArray,
    X: NDArray,
    target_vol: float = 0.15,
    period: int = 252,
    w: int = 21,
    max_leverage: float = 5.0,
) -> NDArray:
    r""" Causal volatility-targeting leverage series for a multi-asset book.

    Multi-asset counterpart of :func:`vol_target`: scales a whole position
    book so its *own* trailing realized volatility targets ``target_vol``,
    instead of scaling a single price series.

    Parameters
    ----------
    W : array_like
        Weights held at each step, shape ``(T, N)`` (e.g. the ``w_mat``
        returned by :func:`fynance.portfolio.allocation.rolling_allocation`).
        A 1-D input is reshaped to ``(T, 1)``.
    X : array_like
        Price/level panel, shape ``(T, N)``, same convention as ``vol_target``
        (prices, not returns). A 1-D input is reshaped to ``(T, 1)``.
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
        Leverage series, shape ``(T,)`` (0 where volatility is not yet
        defined).

    Raises
    ------
    ValueError
        If ``W`` and ``X`` do not share the same shape once 1-D inputs have
        been reshaped to ``(T, 1)``.

    Notes
    -----
    Strictly causal, no-lookahead construction of the book:

    .. math::
        r_t = X_t / X_{t-1} - 1, \quad r_0 = 0

    .. math::
        rb_t = \sum_i W_{t-1, i} \cdot r_{t, i}, \quad rb_0 = 0

    i.e. the weights decided at ``t - 1`` (the last full step of information
    available before ``t``) earn the asset returns realized over
    ``(t - 1, t]`` — the same convention as the training/holding split of
    :func:`fynance.portfolio.allocation.rolling_allocation`. The book level
    ``L = 100 * cumprod(1 + rb)`` is then fed to
    :func:`fynance.features.indicators.realized_volatility` and the leverage
    is derived and clipped exactly as in :func:`vol_target`.

    With a single asset (``N = 1``) and constant unit weight, ``rb`` reduces
    to the asset's own returns, so ``book_vol_target`` reduces to
    ``vol_target`` on the same price series.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[100.0, 100.0],
    ...               [102.0,  98.0],
    ...               [104.0,  96.5],
    ...               [101.0,  99.0],
    ...               [105.0, 101.0]])
    >>> W = np.full((5, 2), 0.5)
    >>> lev = book_vol_target(W, X, target_vol=0.10, w=2, max_leverage=3.0)
    >>> lev.shape
    (5,)
    >>> lev[0], lev[1]
    (0.0, 0.0)
    >>> bool((lev >= 0.0).all() and (lev <= 3.0).all())
    True

    See Also
    --------
    vol_target : single-asset causal volatility-targeting leverage.

    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if W.shape != X.shape:
        raise ValueError(
            f"W and X must share the same shape (T, N) once 1-D inputs are "
            f"reshaped to (T, 1); got W.shape={W.shape} and X.shape={X.shape}."
        )

    r = np.zeros_like(X)
    r[1:] = X[1:] / X[:-1] - 1.0
    rb = np.zeros(X.shape[0])
    rb[1:] = np.sum(W[:-1] * r[1:], axis=1)
    L = 100.0 * np.cumprod(1.0 + rb)

    vol = np.asarray(realized_volatility(L, w=w, period=period))
    with np.errstate(divide='ignore', invalid='ignore'):
        lev = np.where(vol > 0, target_vol / vol, 0.0)

    return np.clip(lev, 0.0, max_leverage)


def transaction_cost(
    weights: NDArray, fee: float = 0.001, axis: int = 0,
) -> NDArray:
    r""" Per-step transaction cost from weight turnover.

    Cost at each step is ``fee`` times the traded amount (turnover):
    :math:`c_t = fee \cdot \sum_i |w_{t,i} - w_{t-1,i}|`, with the first step
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
