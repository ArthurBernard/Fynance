#!/usr/bin/env python3
# coding: utf-8

r""" Rebalancing policies, lot discretization and execution delay.

Composable, strictly causal ``(T, N)`` transforms that sit between an
allocator/signal (the *target* book) and
:func:`fynance.backtest.engine.backtest` (the *effective* book actually
held). A real portfolio does not snap to its target every bar: between
trades the held weights **drift** with asset returns, and trading is
throttled — on a calendar, inside a no-trade band, or under a turnover
budget. These overlays turn a frictionless target series into the book a
desk would actually carry, so a backtest charges turnover on the trades
that really happen rather than on the allocator's ideal (and far busier)
target.

All policies share one drift law: a position held from ``t`` to ``t + 1``
has its weight rescaled by its own gross return relative to the book's,

.. math::
    w_{t+1,i} = \frac{w_{t,i}\,(1 + r_{t+1,i})}
                     {1 + \sum_j w_{t,j}\, r_{t+1,j}},
    \quad r_{t+1,i} = X_{t+1,i} / X_{t,i} - 1,

i.e. the mark-to-market evolution of each position's share of the book
value. If the book return is ``<= -1`` (the book is wiped out) the drifted
weights are held at ``0`` from that bar on. Every transform is causal (row
``t`` uses only ``X`` and ``W`` up to row ``t``), promotes a 1-D input to a
single column, validates float64, and supports long-short books.

Main entry points
-----------------
- :func:`rebalance_calendar` — trade to target on a fixed bar schedule,
  drift between.
- :func:`rebalance_band` — trade only when the drift leaves a no-trade band
  around the target (to the target, or just to the band edge).
- :func:`rebalance_turnover_cap` — move toward the target each bar under a
  per-bar turnover budget.
- :func:`discretize` — round a target book to whole lots given a capital
  base, suppressing trades below a minimum notional.
- :func:`delay` — shift the whole book by a fixed number of bars
  (generalizes the engine's one-bar execution shift).

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = [
    'delay',
    'discretize',
    'rebalance_band',
    'rebalance_calendar',
    'rebalance_turnover_cap',
]


# =========================================================================== #
#                                validation                                   #
# =========================================================================== #


def _as_TN(a: NDArray, name: str) -> NDArray[np.float64]:
    """ Coerce `a` to a float64 ``(T, N)`` array (1-D promoted to ``(T, 1)``). """
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got ndim={arr.ndim}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf).")

    return arr


def _validate_WX(W: NDArray, X: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """ Validate a ``(W, X)`` pair and report whether the input was 1-D.

    Both are coerced to float64 ``(T, N)`` (a 1-D input promoted to
    ``(T, 1)``) and must share the same shape. The returned boolean is
    ``True`` when the original `W` was 1-D, so callers can squeeze the
    output back to ``(T,)``.

    """
    squeeze = np.asarray(W).ndim == 1
    W2 = _as_TN(W, "W")
    X2 = _as_TN(X, "X")
    if W2.shape != X2.shape:
        raise ValueError(
            f"W and X must share the same shape (T, N) once 1-D inputs are "
            f"reshaped to (T, 1); got W.shape={W2.shape} and X.shape={X2.shape}."
        )

    return W2, X2, squeeze


def _returns(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Simple per-asset returns ``r_t = X_t / X_{t-1} - 1`` with ``r_0 = 0``. """
    R = np.zeros_like(X)
    R[1:] = X[1:] / X[:-1] - 1.0

    return R


# =========================================================================== #
#                            drift kernel (shared)                           #
# =========================================================================== #


@njit(cache=True)
def _drift_step(w: NDArray[np.float64], r: NDArray[np.float64]) -> NDArray[np.float64]:
    r""" Mark-to-market drift of a held weight vector over one bar.

    Holding weights `w` and earning per-asset returns `r` over the bar, the
    new weights are each position's updated value as a share of the updated
    book value:

    .. math::
        w^+_i = w_i (1 + r_i) / (1 + \sum_j w_j r_j).

    If the book return :math:`\sum_j w_j r_j` is ``<= -1`` (book wiped out)
    the guard returns an all-zero vector, which stays zero under any further
    drift.
    """
    n = w.shape[0]
    book_ret = 0.0
    for i in range(n):
        book_ret += w[i] * r[i]

    out = np.zeros(n)
    denom = 1.0 + book_ret
    if denom <= 0.0:

        return out

    for i in range(n):
        out[i] = w[i] * (1.0 + r[i]) / denom

    return out


# =========================================================================== #
#                              calendar schedule                             #
# =========================================================================== #


@njit(cache=True)
def _calendar_kernel(
    W: NDArray[np.float64], R: NDArray[np.float64], every: int,
) -> NDArray[np.float64]:
    """ Scan: reset to ``W[t]`` when ``t % every == 0``, drift otherwise. """
    T, N = W.shape
    E = np.empty((T, N))
    for i in range(N):
        E[0, i] = W[0, i]

    for t in range(1, T):
        drift = _drift_step(E[t - 1], R[t])
        if t % every == 0:
            for i in range(N):
                E[t, i] = W[t, i]

        else:
            for i in range(N):
                E[t, i] = drift[i]

    return E


def rebalance_calendar(W: NDArray, X: NDArray, every: int = 21) -> NDArray[np.float64]:
    r""" Rebalance to target weights on a fixed calendar, drift between.

    The effective book is reset to the target ``W[t]`` at every bar ``t``
    with ``t % every == 0`` (bar 0 is always a rebalance bar) and left to
    drift with asset returns on all other bars. ``every = 1`` reproduces the
    raw target series (rebalance every bar); larger periods trade less and
    let the book wander further from target between trades.

    Parameters
    ----------
    W : array_like
        Target weights, shape ``(T, N)`` (a 1-D input is promoted to
        ``(T, 1)`` and the output squeezed back to ``(T,)``). Long-short
        books are supported.
    X : array_like
        Price/level panel aligned with `W`, same shape. Used only to drift
        held weights between rebalances (see :func:`_drift_step`).
    every : int, optional
        Rebalancing period in bars; must be ``>= 1``. Default 21.

    Returns
    -------
    np.ndarray
        Effective weights actually held, same shape as `W`.

    Raises
    ------
    ValueError
        If `W` and `X` do not share the same shape, contain non-finite
        values, or ``every < 1``.

    See Also
    --------
    rebalance_band
    rebalance_turnover_cap

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    >>> X = np.array([[100.0, 100.0], [110.0, 90.0], [121.0, 81.0]])
    >>> np.round(rebalance_calendar(W, X, every=2), 3)
    array([[0.5 , 0.5 ],
           [0.55, 0.45],
           [0.5 , 0.5 ]])

    """
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}.")

    W2, X2, squeeze = _validate_WX(W, X)
    E = _calendar_kernel(W2, _returns(X2), int(every))

    return E.ravel() if squeeze else E


# =========================================================================== #
#                               no-trade band                                #
# =========================================================================== #


@njit(cache=True)
def _band_kernel(
    W: NDArray[np.float64], R: NDArray[np.float64], band: float, edge: bool,
) -> NDArray[np.float64]:
    """ Scan: trade only when the drift's max deviation from target exceeds `band`. """
    T, N = W.shape
    E = np.empty((T, N))
    for i in range(N):
        E[0, i] = W[0, i]

    for t in range(1, T):
        drift = _drift_step(E[t - 1], R[t])
        max_dev = 0.0
        for i in range(N):
            dev = abs(drift[i] - W[t, i])
            if dev > max_dev:
                max_dev = dev

        if max_dev > band:
            if edge:
                for i in range(N):
                    lo = W[t, i] - band
                    hi = W[t, i] + band
                    v = drift[i]
                    if v < lo:
                        v = lo

                    elif v > hi:
                        v = hi

                    E[t, i] = v

            else:
                for i in range(N):
                    E[t, i] = W[t, i]

        else:
            for i in range(N):
                E[t, i] = drift[i]

    return E


def rebalance_band(
    W: NDArray, X: NDArray, band: float = 0.05, mode: str = 'full',
) -> NDArray[np.float64]:
    r""" Rebalance only when the drift leaves a no-trade band around the target.

    Each bar the held book is drifted with asset returns and compared to the
    current target; a trade is triggered only when the largest per-asset
    deviation exceeds `band`,
    :math:`\max_i |w^{\text{drift}}_{t,i} - W_{t,i}| > band`. When it does:

    - ``mode='full'`` trades all the way back to the target ``W[t]``;
    - ``mode='edge'`` trades each asset only to the near band edge, i.e.
      clips the drifted weight to ``[W[t] - band, W[t] + band]`` — the
      breaching asset lands exactly on the boundary and the book stays
      inside the band while trading as little as possible.

    Bar 0 always establishes the full target book; the band governs bars
    ``>= 1``.

    Parameters
    ----------
    W : array_like
        Target weights, shape ``(T, N)`` (1-D promoted to ``(T, 1)``,
        output squeezed back). Long-short books are supported.
    X : array_like
        Price/level panel aligned with `W`, same shape.
    band : float, optional
        No-trade half-width around each target weight; must be ``>= 0``.
        Default 0.05.
    mode : {'full', 'edge'}, optional
        Whether a triggered trade goes to the target (``'full'``, default)
        or only to the band edge (``'edge'``).

    Returns
    -------
    np.ndarray
        Effective weights actually held, same shape as `W`.

    Raises
    ------
    ValueError
        If `W` and `X` do not share the same shape, contain non-finite
        values, ``band < 0``, or `mode` is not ``'full'`` / ``'edge'``.

    See Also
    --------
    rebalance_calendar
    rebalance_turnover_cap

    Examples
    --------
    A 20% one-day divergence breaks a 5% band; ``'full'`` snaps back to
    target while ``'edge'`` stops on the band boundary:

    >>> import numpy as np
    >>> W = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> X = np.array([[100.0, 100.0], [120.0, 80.0]])
    >>> rebalance_band(W, X, band=0.05, mode='full')
    array([[0.5, 0.5],
           [0.5, 0.5]])
    >>> rebalance_band(W, X, band=0.05, mode='edge')
    array([[0.5 , 0.5 ],
           [0.55, 0.45]])

    """
    if mode not in ('full', 'edge'):
        raise ValueError(f"Unknown mode {mode!r}; expected 'full' or 'edge'.")

    if band < 0:
        raise ValueError(f"band must be >= 0, got {band}.")

    W2, X2, squeeze = _validate_WX(W, X)
    E = _band_kernel(W2, _returns(X2), float(band), mode == 'edge')

    return E.ravel() if squeeze else E


# =========================================================================== #
#                              turnover budget                               #
# =========================================================================== #


@njit(cache=True)
def _turnover_cap_kernel(
    W: NDArray[np.float64], R: NDArray[np.float64], budget: float,
) -> NDArray[np.float64]:
    """ Scan: move toward the target each bar under ``sum_i |dw_i| <= budget``. """
    T, N = W.shape
    E = np.empty((T, N))
    prev = np.zeros(N)
    for t in range(T):
        if t == 0:
            drift = np.zeros(N)

        else:
            drift = _drift_step(prev, R[t])

        desired = 0.0
        for i in range(N):
            desired += abs(W[t, i] - drift[i])

        if desired <= budget or desired == 0.0:
            for i in range(N):
                E[t, i] = W[t, i]

        else:
            scale = budget / desired
            for i in range(N):
                E[t, i] = drift[i] + scale * (W[t, i] - drift[i])

        for i in range(N):
            prev[i] = E[t, i]

    return E


def rebalance_turnover_cap(
    W: NDArray, X: NDArray, budget: float = 0.10,
) -> NDArray[np.float64]:
    r""" Move toward the target each bar under a per-bar turnover budget.

    Each bar the held book is drifted with asset returns, then traded toward
    the current target ``W[t]`` — but the trade
    :math:`\Delta w = E_t - w^{\text{drift}}_t` is capped so its one-way
    turnover :math:`\sum_i |\Delta w_i|` never exceeds `budget`. When the
    desired move is larger than the budget the whole trade vector is scaled
    down by ``budget / desired`` (its direction preserved, only its size
    shrunk), so the book eases toward a persistent target over several bars
    instead of snapping in one. The book starts flat, so the initial entry
    is throttled by the same budget.

    Parameters
    ----------
    W : array_like
        Target weights, shape ``(T, N)`` (1-D promoted to ``(T, 1)``,
        output squeezed back). Long-short books are supported.
    X : array_like
        Price/level panel aligned with `W`, same shape.
    budget : float, optional
        Maximum one-way turnover ``sum_i |dw_i|`` allowed per bar; must be
        ``>= 0``. Default 0.10.

    Returns
    -------
    np.ndarray
        Effective weights actually held, same shape as `W`.

    Raises
    ------
    ValueError
        If `W` and `X` do not share the same shape, contain non-finite
        values, or ``budget < 0``.

    See Also
    --------
    rebalance_calendar
    rebalance_band

    Examples
    --------
    A 50% budget lets an all-in-asset-0 to all-in-asset-1 target ease over
    several bars; the first bar can move at most 0.5 of turnover from flat:

    >>> import numpy as np
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> X = np.array([[100.0, 100.0], [100.0, 100.0]])
    >>> E = rebalance_turnover_cap(W, X, budget=0.5)
    >>> np.round(E[0], 3)
    array([0.5, 0. ])

    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}.")

    W2, X2, squeeze = _validate_WX(W, X)
    E = _turnover_cap_kernel(W2, _returns(X2), float(budget))

    return E.ravel() if squeeze else E


# =========================================================================== #
#                            lot discretization                              #
# =========================================================================== #


@njit(cache=True)
def _discretize_kernel(
    W: NDArray[np.float64],
    P: NDArray[np.float64],
    capital: float,
    lot: float,
    min_notional: float,
) -> NDArray[np.float64]:
    """ Scan: round each target notional to whole lots, suppress tiny trades. """
    T, N = W.shape
    E = np.empty((T, N))
    prev = np.zeros(N)
    for t in range(T):
        for i in range(N):
            price = P[t, i]
            target_shares = W[t, i] * capital / price
            rounded = np.round(target_shares / lot) * lot
            trade_notional = abs(rounded - prev[i]) * price
            if trade_notional < min_notional:
                shares = prev[i]

            else:
                shares = rounded

            E[t, i] = shares * price / capital
            prev[i] = shares

    return E


def discretize(
    W: NDArray,
    prices: NDArray,
    capital: float = 1e6,
    lot: float = 1.0,
    min_notional: float = 0.0,
) -> NDArray[np.float64]:
    r""" Round a target weight book to whole lots on a fixed capital base.

    Turns continuous target weights into the book an integer-lot execution
    would actually hold. At each bar the target notional ``W[t, i] *
    capital`` is converted to shares at ``prices[t, i]``, rounded to the
    nearest multiple of `lot`, and converted back to a weight
    ``shares * price / capital``. A rebalancing trade whose notional
    ``|shares_new - shares_prev| * price`` falls below `min_notional` is
    suppressed (the previous share count is kept), which removes the churn
    of tiny odd-lot adjustments. The scan carries the held share count
    across bars, so the output depends on trade *history*, not only on the
    current target.

    Parameters
    ----------
    W : array_like
        Target weights, shape ``(T, N)`` (1-D promoted to ``(T, 1)``,
        output squeezed back). Long-short books are supported (negative
        weights round to negative share counts).
    prices : array_like
        Strictly positive price levels aligned with `W`, same shape.
    capital : float, optional
        Reference capital defining the notional base; must be ``> 0``.
        Default ``1e6``.
    lot : float, optional
        Tradeable lot size in shares (e.g. ``100`` for round lots); must be
        ``> 0``. Default 1.0.
    min_notional : float, optional
        Trades whose notional value is strictly below this threshold are
        skipped, keeping the previous position; must be ``>= 0``. Default
        0.0 (never skip).

    Returns
    -------
    np.ndarray
        Effective weights implied by the rounded share book, same shape as
        `W`.

    Raises
    ------
    ValueError
        If `W` and `prices` do not share the same shape, contain non-finite
        values, or if ``capital <= 0``, ``lot <= 0`` or ``min_notional < 0``.

    Examples
    --------
    With ``capital=1000`` and unit lots, a 50% target on a $300 asset buys
    ``round(500 / 300) = 2`` shares, i.e. an effective 60% weight; the $50
    asset lands exactly on 50%:

    >>> import numpy as np
    >>> W = np.array([[0.5, 0.5]])
    >>> prices = np.array([[300.0, 50.0]])
    >>> discretize(W, prices, capital=1000.0, lot=1.0)
    array([[0.6, 0.5]])

    """
    if capital <= 0:
        raise ValueError(f"capital must be > 0, got {capital}.")

    if lot <= 0:
        raise ValueError(f"lot must be > 0, got {lot}.")

    if min_notional < 0:
        raise ValueError(f"min_notional must be >= 0, got {min_notional}.")

    W2, P2, squeeze = _validate_WX(W, prices)
    E = _discretize_kernel(W2, P2, float(capital), float(lot), float(min_notional))

    return E.ravel() if squeeze else E


# =========================================================================== #
#                              execution delay                               #
# =========================================================================== #


def delay(W: NDArray, steps: int = 1) -> NDArray[np.float64]:
    r""" Shift a weight book forward by a fixed number of bars.

    Delays execution by `steps` bars: row ``t`` of the output is the target
    of row ``t - steps`` (the first `steps` rows are zero, i.e. flat). This
    generalizes the one-bar causal shift the backtest engine applies by
    default (``shift=True``) — use it to model a longer decision-to-fill lag,
    or to pre-shift a book fed to ``backtest(..., shift=False)``.

    Parameters
    ----------
    W : array_like
        Weight book, shape ``(T, N)`` (1-D promoted to ``(T, 1)``, output
        squeezed back). Long-short books are supported.
    steps : int, optional
        Number of bars to delay; must be ``>= 0``. ``steps=0`` returns a
        copy unchanged; ``steps >= T`` returns an all-zero book. Default 1.

    Returns
    -------
    np.ndarray
        The shifted book, same shape as `W`.

    Raises
    ------
    ValueError
        If `W` is not 1-D or 2-D, contains non-finite values, or
        ``steps < 0``.

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    >>> delay(W, steps=1)
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 0. ]])

    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")

    squeeze = np.asarray(W).ndim == 1
    W2 = _as_TN(W, "W")
    E = np.zeros_like(W2)
    if steps < W2.shape[0]:
        E[steps:] = W2[:W2.shape[0] - steps]

    return E.ravel() if squeeze else E
