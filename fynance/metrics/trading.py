#!/usr/bin/env python3
# coding: utf-8

""" Trading-profile metrics — churn of a position series or book.

These describe *how* a strategy trades (turnover of direction) rather than how
its equity curve performs, so — like :func:`information_coefficient` — they take
a **position** series, not an equity curve, and are intentionally kept out of
the ``METRICS`` registry.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = [
    'annual_turnover',
    'exposure_summary',
    'gross_exposure',
    'net_exposure',
    'sign_changes',
    'trades_per_year',
    'turnover_series',
]


def sign_changes(positions: NDArray, *, axis: int = 0) -> NDArray | int:
    r""" Number of position sign changes (long <-> flat <-> short).

    Counts the steps where ``sign(pos_t) != sign(pos_{t-1})`` along the time
    ``axis`` — the round-trip churn a turnover-blind ``total_cost`` hides. Flat
    (``0``) is a distinct state, so ``long -> flat`` and ``flat -> short`` each
    count as one change. Pairs straddling a ``NaN`` are not counted.

    Parameters
    ----------
    positions : array_like
        Position / weight series, shape ``(T,)`` or ``(T, n_assets)``.
    axis : int, optional
        Time axis. Default 0.

    Returns
    -------
    int or numpy.ndarray
        Total count for a 1-D series; a per-asset count vector for a 2-D book.

    Examples
    --------
    >>> import numpy as np
    >>> sign_changes(np.array([1.0, 1.0, -1.0, -1.0, 0.0, 1.0]))
    3
    >>> sign_changes(np.array([[1.0, 0.0], [-1.0, 0.0], [-1.0, 1.0]]))
    array([1, 1])

    """
    s = np.sign(np.asarray(positions, dtype=np.float64))
    t = s.shape[axis]

    if t < 2:

        return 0 if s.ndim == 1 else np.zeros(s.shape[1 - axis], dtype=int)

    prev = np.take(s, np.arange(t - 1), axis=axis)
    cur = np.take(s, np.arange(1, t), axis=axis)
    changed = (prev != cur) & ~(np.isnan(prev) | np.isnan(cur))
    out = changed.sum(axis=axis)

    return int(out) if np.ndim(out) == 0 else out.astype(int)


def trades_per_year(positions: NDArray, period: int = 252, *,
                    axis: int = 0) -> NDArray | float:
    r""" Annualized number of position sign changes.

    :func:`sign_changes` scaled to a yearly rate, ``n_changes / T * period``, so
    two strategies sampled at different frequencies stay comparable.

    Parameters
    ----------
    positions : array_like
        Position / weight series, shape ``(T,)`` or ``(T, n_assets)``.
    period : int, optional
        Annualization factor (bars per year, 252 for daily). Default 252.
    axis : int, optional
        Time axis. Default 0.

    Returns
    -------
    float or numpy.ndarray
        Annualized rate for a 1-D series; a per-asset vector for a 2-D book.

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.array([1.0, -1.0, 1.0, -1.0])  # flips direction every bar
    >>> float(trades_per_year(pos, period=252))
    189.0

    """
    p = np.asarray(positions, dtype=np.float64)
    t = p.shape[axis]
    sc = sign_changes(p, axis=axis)

    if t < 1:

        return 0.0 if p.ndim == 1 else np.zeros(p.shape[1 - axis])

    rate = np.asarray(sc, dtype=np.float64) / t * period

    return float(rate) if np.ndim(rate) == 0 else rate


def _as_book(W: NDArray) -> NDArray:
    """ Coerce to a 2-D ``(T, N)`` book (a 1-D series becomes a single column). """
    w = np.asarray(W, dtype=np.float64)

    return w if w.ndim == 2 else w.reshape(w.shape[0], -1)


def turnover_series(W: NDArray) -> NDArray:
    r""" One-way turnover per bar, :math:`\sum_i |w_{t,i} - w_{t-1,i}|`.

    The first bar charges the initial position from flat (:math:`|w_{0,i}|`),
    same convention as :func:`fynance.portfolio.sizing.transaction_cost` — in
    fact ``turnover_series(W) * fee == transaction_cost(W, fee)`` exactly,
    :func:`turnover_series` is just the fee-free churn underneath it.

    Parameters
    ----------
    W : array_like
        Weights held at each step, shape ``(T,)`` or ``(T, N)``. A 1-D input
        is reshaped to ``(T, 1)``.

    Returns
    -------
    numpy.ndarray
        Turnover per bar, shape ``(T,)``.

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, -1.0], [0.0, 0.0]])
    >>> turnover_series(W)
    array([1., 1., 2., 2.])

    See Also
    --------
    annual_turnover : same churn, annualized to a single rate.
    fynance.portfolio.sizing.transaction_cost : the fee-weighted counterpart.

    """
    w = _as_book(W)
    turnover = np.empty(w.shape[0])
    turnover[0] = np.abs(w[0]).sum()
    turnover[1:] = np.abs(np.diff(w, axis=0)).sum(axis=1)

    return turnover


def annual_turnover(W: NDArray, period: int = 252) -> float:
    r""" Annualized one-way turnover, ``mean(turnover_series(W)) * period``.

    Parameters
    ----------
    W : array_like
        Weights held at each step, shape ``(T,)`` or ``(T, N)``. A 1-D input
        is reshaped to ``(T, 1)``.
    period : int, optional
        Annualization factor (bars per year, 252 for daily). Default 252.

    Returns
    -------
    float
        Annualized turnover rate.

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, -1.0], [0.0, 0.0]])
    >>> annual_turnover(W, period=252)
    378.0

    See Also
    --------
    turnover_series : the underlying per-bar turnover.

    """
    return float(np.mean(turnover_series(W)) * period)


def gross_exposure(W: NDArray) -> NDArray:
    r""" Gross exposure per bar, :math:`\sum_i |w_{t,i}|` (total book leverage).

    Parameters
    ----------
    W : array_like
        Weights held at each step, shape ``(T,)`` or ``(T, N)``. A 1-D input
        is reshaped to ``(T, 1)``.

    Returns
    -------
    numpy.ndarray
        Gross exposure per bar, shape ``(T,)``.

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, -1.0], [0.0, 0.0]])
    >>> gross_exposure(W)
    array([1., 1., 2., 0.])

    See Also
    --------
    net_exposure : the signed (long/short bias) counterpart.

    """
    return np.abs(_as_book(W)).sum(axis=1)


def net_exposure(W: NDArray) -> NDArray:
    r""" Net exposure per bar, :math:`\sum_i w_{t,i}` (long/short bias).

    Parameters
    ----------
    W : array_like
        Weights held at each step, shape ``(T,)`` or ``(T, N)``. A 1-D input
        is reshaped to ``(T, 1)``.

    Returns
    -------
    numpy.ndarray
        Net exposure per bar, shape ``(T,)``. Positive is net long, negative
        net short, zero flat (fully hedged or no position).

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, -1.0], [0.0, 0.0]])
    >>> net_exposure(W)
    array([ 1.,  0., -2.,  0.])

    See Also
    --------
    gross_exposure : the unsigned (total leverage) counterpart.

    """
    return _as_book(W).sum(axis=1)


def exposure_summary(W: NDArray, period: int = 252) -> dict:
    r""" One-shot summary of a book's turnover and exposure.

    Combines :func:`annual_turnover`, :func:`gross_exposure` and
    :func:`net_exposure` into the small set of numbers that describe how a
    book trades (churn) and sits (leverage, long/short bias) — the position
    -level analogue of :func:`fynance.metrics.summary.summary` for an equity
    curve.

    Parameters
    ----------
    W : array_like
        Weights held at each step, shape ``(T,)`` or ``(T, N)``. A 1-D input
        is reshaped to ``(T, 1)``.
    period : int, optional
        Annualization factor (bars per year, 252 for daily). Default 252.

    Returns
    -------
    dict
        ``{'annual_turnover', 'mean_gross', 'max_gross', 'mean_net',
        'min_net', 'max_net', 'pct_long', 'pct_short', 'pct_flat'}``; the
        ``pct_*`` entries are the percentage of bars with net exposure
        ``> 0``, ``< 0`` and ``== 0`` respectively (they sum to 100).

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, -1.0], [0.0, 0.0]])
    >>> out = exposure_summary(W, period=252)
    >>> out['annual_turnover'], out['mean_gross'], out['mean_net']
    (378.0, 1.0, -0.25)
    >>> out['pct_long'], out['pct_short'], out['pct_flat']
    (25.0, 25.0, 50.0)

    See Also
    --------
    annual_turnover : the churn entry alone.
    gross_exposure : the per-bar gross exposure series.
    net_exposure : the per-bar net exposure series.

    """
    gross = gross_exposure(W)
    net = net_exposure(W)
    n = net.shape[0]

    return {
        'annual_turnover': annual_turnover(W, period=period),
        'mean_gross': float(np.mean(gross)),
        'max_gross': float(np.max(gross)),
        'mean_net': float(np.mean(net)),
        'min_net': float(np.min(net)),
        'max_net': float(np.max(net)),
        'pct_long': float(np.sum(net > 0) / n * 100.0),
        'pct_short': float(np.sum(net < 0) / n * 100.0),
        'pct_flat': float(np.sum(net == 0) / n * 100.0),
    }
