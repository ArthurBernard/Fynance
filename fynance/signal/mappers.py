#!/usr/bin/env python3
# coding: utf-8

""" Signal mappers: turn a model prediction into a position.

Pure numpy and causal: the position at ``t`` depends only on the prediction at
``t`` (the engine in :mod:`fynance.backtest` applies the causal one-step shift).

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.algorithms.sizing import vol_target

__all__ = ['sign', 'threshold', 'rank', 'vol_target_position']


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
        Number of assets in the long and short legs.

    Returns
    -------
    numpy.ndarray
        Weights of shape ``(T, n_assets)``.

    """
    p = np.asarray(pred, dtype=np.float64)

    if p.ndim != 2:

        raise ValueError("rank expects a 2-D (T, n_assets) prediction")

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
    :func:`fynance.algorithms.sizing.vol_target` (inverse of *past* realized
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
