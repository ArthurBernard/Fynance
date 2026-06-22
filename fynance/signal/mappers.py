#!/usr/bin/env python3
# coding: utf-8

""" Signal mappers: turn a model prediction into a position.

Pure numpy and causal: the position at ``t`` depends only on the prediction at
``t`` (the engine in :mod:`fynance.backtest` applies the causal one-step shift).

The **anti-churn** mappers — :func:`ema_smooth`, :func:`deadband`,
:func:`min_hold` — are stateful (each output depends on the past output) but
strictly causal; compose them to cut turnover where transaction costs would
otherwise eat the edge (high fees / high frequency).

Initial-state convention: the stateful mappers seed their running state from
the first input (``out[0] == pred[0]``) rather than from zero, so the first
bar is passed through unchanged and the smoothing/holding logic starts from a
realistic position rather than an artificial flat one.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.portfolio.sizing import vol_target

__all__ = ['sign', 'threshold', 'rank', 'vol_target_position',
           'ema_smooth', 'deadband', 'min_hold']


def sign(pred: NDArray) -> NDArray[np.float64]:
    """ Long/short by the sign of the prediction (``+1`` / ``-1`` / ``0``).

    Examples
    --------
    >>> import numpy as np
    >>> sign(np.array([0.3, -0.1, 0.0]))
    array([ 1., -1.,  0.])

    """
    return np.sign(np.asarray(pred, dtype=np.float64))


def threshold(
    pred: NDArray,
    long: float = 0.0,
    short: float = 0.0,
) -> NDArray[np.float64]:
    """ Position with a flat dead-band.

    ``+1`` where ``pred > long``, ``-1`` where ``pred < short``, ``0`` between.

    Examples
    --------
    >>> import numpy as np
    >>> threshold(np.array([0.5, 0.05, -0.5]), long=0.1, short=-0.1)
    array([ 1.,  0., -1.])

    """
    p = np.asarray(pred, dtype=np.float64)
    pos = np.zeros_like(p)
    pos[p > long] = 1.0
    pos[p < short] = -1.0

    return pos


def rank(pred: NDArray, top: int, bottom: int) -> NDArray[np.float64]:
    """ Cross-sectional long/short by rank.

    For each time step (row of a 2-D ``(T, n_assets)`` prediction), go long the
    ``top`` highest-ranked assets and short the ``bottom`` lowest, equally
    weighted within each leg (dollar-neutral).

    Parameters
    ----------
    pred : array-like, shape (T, n_assets)
        Cross-sectional predictions.
    top, bottom : int
        Number of assets in the long and short legs. Both must be ``>= 0`` and
        ``top + bottom`` must not exceed ``n_assets`` -- otherwise the legs
        would overlap and the long assignment would silently overwrite the
        short, breaking dollar-neutrality.

    Returns
    -------
    numpy.ndarray
        Weights of shape ``(T, n_assets)``.

    Raises
    ------
    ValueError
        If ``pred`` is not 2-D, if ``top`` or ``bottom`` is negative, or if
        ``top + bottom`` exceeds the number of assets.

    Examples
    --------
    >>> import numpy as np
    >>> w = rank(np.array([[1.0, 2.0, 3.0, 4.0]]), top=1, bottom=1)
    >>> w.sum()  # dollar-neutral
    0.0

    """
    p = np.asarray(pred, dtype=np.float64)

    if p.ndim != 2:

        raise ValueError("rank expects a 2-D (T, n_assets) prediction")

    if top < 0 or bottom < 0:

        raise ValueError(
            f"top and bottom must be >= 0, got top={top}, bottom={bottom}"
        )

    n_assets = p.shape[1]

    if top + bottom > n_assets:

        raise ValueError(
            f"overlapping legs: top + bottom = {top + bottom} > "
            f"n_assets = {n_assets}; the long and short legs would overlap"
        )

    weights = np.zeros_like(p)
    order = np.argsort(p, axis=1)

    for t in range(p.shape[0]):
        if bottom > 0:
            weights[t, order[t, :bottom]] = -1.0 / bottom

        if top > 0:
            weights[t, order[t, -top:]] = 1.0 / top

    return weights


def vol_target_position(
    signal: NDArray,
    prices: NDArray,
    target_vol: float = 0.15,
    period: int = 252,
    w: int = 21,
    max_leverage: float = 5.0,
) -> NDArray[np.float64]:
    """ Scale a directional signal by causal volatility-targeting leverage.

    Multiplies a directional ``signal`` by the leverage from
    :func:`fynance.portfolio.sizing.vol_target` (inverse of *past* realized
    volatility). Strictly causal.
    """
    sig = np.asarray(signal, dtype=np.float64)
    lev = vol_target(
        prices,
        target_vol=target_vol,
        period=period,
        w=w,
        max_leverage=max_leverage,
    )

    return sig * lev


def ema_smooth(pred: NDArray, alpha: float = 0.2) -> NDArray[np.float64]:
    """ Causally smooth a position with an exponential moving average.

    ``out[t] = alpha * pred[t] + (1 - alpha) * out[t-1]`` — a low ``alpha`` reacts
    slowly, cutting turnover by damping minute-to-minute flips. Anti-churn.

    Parameters
    ----------
    pred : array-like, shape (T,)
        Raw position / prediction series.
    alpha : float
        Smoothing factor in ``(0, 1]`` (``1`` = no smoothing).

    Examples
    --------
    >>> import numpy as np
    >>> ema_smooth(np.array([0.0, 1.0, 1.0, -1.0]), alpha=0.5)
    array([ 0.   ,  0.5  ,  0.75 , -0.125])

    """
    if not 0.0 < alpha <= 1.0:

        raise ValueError("alpha must be in (0, 1]")

    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    out = np.empty_like(p)
    acc = 0.0 if p.size == 0 else p[0]
    for t in range(p.size):
        acc = alpha * p[t] + (1.0 - alpha) * acc
        out[t] = acc

    return out


def deadband(pred: NDArray, band: float = 0.1) -> NDArray[np.float64]:
    """ Hold the previous position unless the target moves by more than ``band``.

    A *sticky* dead-band: ``out[t] = pred[t]`` only when
    ``|pred[t] - out[t-1]| > band``, else ``out[t] = out[t-1]``. Magnitude is
    preserved (unlike :func:`threshold`), but small oscillations are ignored, so
    the position changes only on meaningful moves. Anti-churn.

    Parameters
    ----------
    pred : array-like, shape (T,)
        Raw position / prediction series.
    band : float
        Minimum change required to update the held position (``>= 0``).

    Examples
    --------
    >>> import numpy as np
    >>> deadband(np.array([0.0, 0.05, 0.5, 0.55, -0.4]), band=0.2)
    array([ 0. ,  0. ,  0.5,  0.5, -0.4])

    """
    if band < 0.0:

        raise ValueError("band must be >= 0")

    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    out = np.empty_like(p)
    held = 0.0 if p.size == 0 else p[0]
    for t in range(p.size):
        if abs(p[t] - held) > band:
            held = p[t]
        out[t] = held

    return out


def min_hold(pos: NDArray, hold: int, tol: float = 1e-9) -> NDArray[np.float64]:
    """ Enforce a minimum holding period between position changes.

    Once the position changes, it is **frozen for ``hold`` bars** before another
    change can take effect — a hard cap on trade frequency. Strictly causal.

    Parameters
    ----------
    pos : array-like, shape (T,)
        Position series (e.g. already mapped to a target).
    hold : int
        Minimum number of bars between two changes (``hold <= 1`` is a no-op).
    tol : float
        Changes smaller than ``tol`` are treated as no change.

    Examples
    --------
    >>> import numpy as np
    >>> min_hold(np.array([1., -1., 1., -1., 1., -1.]), hold=3)
    array([ 1.,  1.,  1., -1., -1., -1.])

    """
    p = np.asarray(pos, dtype=np.float64).reshape(-1)
    if hold <= 1 or p.size == 0:

        return p.copy()

    out = np.empty_like(p)
    held = p[0]
    out[0] = held
    since = 0
    for t in range(1, p.size):
        since += 1
        if since >= hold and abs(p[t] - held) > tol:
            held = p[t]
            since = 0
        out[t] = held

    return out
