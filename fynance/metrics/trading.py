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

__all__ = ['sign_changes', 'trades_per_year']


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
