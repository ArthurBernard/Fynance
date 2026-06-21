#!/usr/bin/env python3
# coding: utf-8

""" Multi-series OHLCV technical indicators.

Indicators that need more than a single price series — High/Low (``atr``,
``adx``, ``williams_r``) or Volume (``obv``, ``vwap``). Each takes the raw
aligned arrays (the primary, numpy-idiomatic API) and also accepts an
:class:`~fynance.core.OHLCV` container as the first argument.

All indicators are **causal**: the value at ``t`` uses only ``data[..t]``.
Rolling loops are Numba ``@njit`` kernels living in this module.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray

# Local packages
from fynance.core.ohlcv import OHLCV

__all__ = ['adx', 'atr', 'obv', 'vwap', 'williams_r']


def _as1d(x) -> NDArray[np.float64]:
    """ Coerce an array-like to a contiguous 1-D ``float64`` array. """
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))


def _unpack_hlc(high, low, close):
    """ Resolve (high, low, close) from raw arrays or a leading ``OHLCV``. """
    if isinstance(high, OHLCV):

        return high.high, high.low, high.close

    if low is None or close is None:

        raise ValueError("provide high, low and close (or a single OHLCV)")

    return _as1d(high), _as1d(low), _as1d(close)


# =========================================================================== #
#                              Numba kernels                                  #
# =========================================================================== #


@njit(cache=True)
def _atr_kernel(high, low, close, w):
    """ True range + Wilder-smoothed ATR (causal). """
    n = high.size
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for t in range(1, n):
        hl = high[t] - low[t]
        hc = abs(high[t] - close[t - 1])
        lc = abs(low[t] - close[t - 1])
        tr[t] = max(hl, max(hc, lc))

    atr = np.empty(n)
    atr[0] = tr[0]
    for t in range(1, n):
        atr[t] = (atr[t - 1] * (w - 1) + tr[t]) / w

    return atr


@njit(cache=True)
def _adx_kernel(high, low, close, w):
    """ Wilder ADX (causal): TR/+DM/-DM -> +DI/-DI -> DX -> ADX. """
    n = high.size
    tr = np.empty(n)
    pdm = np.empty(n)
    ndm = np.empty(n)
    tr[0] = high[0] - low[0]
    pdm[0] = 0.0
    ndm[0] = 0.0
    for t in range(1, n):
        up = high[t] - high[t - 1]
        dn = low[t - 1] - low[t]
        pdm[t] = up if (up > dn and up > 0.0) else 0.0
        ndm[t] = dn if (dn > up and dn > 0.0) else 0.0
        hl = high[t] - low[t]
        hc = abs(high[t] - close[t - 1])
        lc = abs(low[t] - close[t - 1])
        tr[t] = max(hl, max(hc, lc))

    str_ = np.empty(n)
    spdm = np.empty(n)
    sndm = np.empty(n)
    str_[0] = tr[0]
    spdm[0] = pdm[0]
    sndm[0] = ndm[0]
    for t in range(1, n):
        str_[t] = (str_[t - 1] * (w - 1) + tr[t]) / w
        spdm[t] = (spdm[t - 1] * (w - 1) + pdm[t]) / w
        sndm[t] = (sndm[t - 1] * (w - 1) + ndm[t]) / w

    dx = np.empty(n)
    for t in range(n):
        denom = str_[t] if str_[t] != 0.0 else 1e-12
        pdi = 100.0 * spdm[t] / denom
        ndi = 100.0 * sndm[t] / denom
        s = pdi + ndi
        dx[t] = 100.0 * abs(pdi - ndi) / (s if s != 0.0 else 1e-12)

    adx = np.empty(n)
    adx[0] = dx[0]
    for t in range(1, n):
        adx[t] = (adx[t - 1] * (w - 1) + dx[t]) / w

    return adx


@njit(cache=True)
def _williams_kernel(high, low, close, w):
    """ Williams %R over a trailing window (causal). """
    n = high.size
    out = np.empty(n)
    for t in range(n):
        a = 0 if t - w + 1 < 0 else t - w + 1
        hh = high[a]
        ll = low[a]
        for j in range(a + 1, t + 1):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        denom = hh - ll
        out[t] = 0.0 if denom == 0.0 else -100.0 * (hh - close[t]) / denom

    return out


@njit(cache=True)
def _obv_kernel(close, volume):
    """ On-Balance Volume (causal cumulative). """
    n = close.size
    out = np.empty(n)
    out[0] = 0.0
    for t in range(1, n):
        if close[t] > close[t - 1]:
            out[t] = out[t - 1] + volume[t]
        elif close[t] < close[t - 1]:
            out[t] = out[t - 1] - volume[t]
        else:
            out[t] = out[t - 1]

    return out


@njit(cache=True)
def _vwap_kernel(tp, volume, w):
    """ VWAP of typical price ``tp`` (cumulative if ``w <= 0``, else rolling). """
    n = tp.size
    out = np.empty(n)
    if w <= 0:
        cpv = 0.0
        cv = 0.0
        for t in range(n):
            cpv += tp[t] * volume[t]
            cv += volume[t]
            out[t] = cpv / cv if cv != 0.0 else np.nan

    else:
        for t in range(n):
            a = 0 if t - w + 1 < 0 else t - w + 1
            spv = 0.0
            sv = 0.0
            for j in range(a, t + 1):
                spv += tp[j] * volume[j]
                sv += volume[j]
            out[t] = spv / sv if sv != 0.0 else np.nan

    return out


# =========================================================================== #
#                            Public indicators                                #
# =========================================================================== #


def atr(high, low=None, close=None, w: int = 14) -> NDArray[np.float64]:
    r""" Average True Range (Wilder), causal.

    The true range :math:`TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)`
    smoothed by a Wilder moving average of window ``w``.

    Parameters
    ----------
    high, low, close : array-like, or high = OHLCV
        High/Low/Close series, or a single :class:`~fynance.core.OHLCV`.
    w : int, optional
        Smoothing window. Default 14.

    Returns
    -------
    numpy.ndarray
        ATR series, ``>= 0``, aligned with the input.

    Examples
    --------
    >>> import numpy as np
    >>> h = np.array([10., 11., 12., 11.5])
    >>> low = np.array([9., 9.5, 11., 10.5])
    >>> c = np.array([9.5, 10.5, 11.5, 11.])
    >>> atr(h, low, c, w=2).round(4)
    array([1.    , 1.25  , 1.375 , 1.1875])

    """
    high, low, close = _unpack_hlc(high, low, close)

    return _atr_kernel(high, low, close, int(w))


def adx(high, low=None, close=None, w: int = 14) -> NDArray[np.float64]:
    r""" Average Directional Index (Wilder), causal.

    Trend-strength oscillator built from the smoothed directional movement
    (+DI / -DI) and their normalized spread (DX), itself Wilder-smoothed.

    Parameters
    ----------
    high, low, close : array-like, or high = OHLCV
        High/Low/Close series, or a single :class:`~fynance.core.OHLCV`.
    w : int, optional
        Smoothing window. Default 14.

    Returns
    -------
    numpy.ndarray
        ADX series in ``[0, 100]``, aligned with the input.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> h = 100 + np.cumsum(np.abs(rng.standard_normal(n)))
    >>> low = h - np.abs(rng.standard_normal(n))
    >>> c = (h + low) / 2
    >>> out = adx(h, low, c, w=14)
    >>> bool(np.all((out >= 0) & (out <= 100)))
    True

    """
    high, low, close = _unpack_hlc(high, low, close)

    return _adx_kernel(high, low, close, int(w))


def williams_r(high, low=None, close=None, w: int = 14) -> NDArray[np.float64]:
    r""" Williams %R over a trailing window, causal.

    :math:`\%R_t = -100 \cdot (HH_t - C_t) / (HH_t - LL_t)`, where ``HH``/``LL``
    are the rolling high/low of window ``w``. Bounded in ``[-100, 0]``.

    Parameters
    ----------
    high, low, close : array-like, or high = OHLCV
        High/Low/Close series, or a single :class:`~fynance.core.OHLCV`.
    w : int, optional
        Look-back window. Default 14.

    Returns
    -------
    numpy.ndarray
        Williams %R in ``[-100, 0]``, aligned with the input.

    Examples
    --------
    >>> import numpy as np
    >>> h = np.array([10., 11., 12., 11.])
    >>> low = np.array([9., 9., 10., 10.])
    >>> c = np.array([9.5, 10.5, 11.5, 10.5])
    >>> williams_r(h, low, c, w=2).round(2)
    array([-50.  , -25.  , -16.67, -75.  ])

    """
    high, low, close = _unpack_hlc(high, low, close)

    return _williams_kernel(high, low, close, int(w))


def obv(close, volume=None) -> NDArray[np.float64]:
    r""" On-Balance Volume, causal.

    Signed cumulative volume: add ``volume`` on up-closes, subtract on
    down-closes, carry on flat closes (starts at ``0``).

    Parameters
    ----------
    close : array-like, or OHLCV
        Close series, or a single :class:`~fynance.core.OHLCV`.
    volume : array-like, optional
        Volume series (required unless ``close`` is an ``OHLCV``).

    Returns
    -------
    numpy.ndarray
        OBV series, aligned with the input.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array([10., 11., 10.5, 10.5, 12.])
    >>> v = np.array([100., 150., 120., 80., 200.])
    >>> obv(c, v)
    array([  0., 150.,  30.,  30., 230.])

    """
    if isinstance(close, OHLCV):
        close, volume = close.close, close.volume

    elif volume is None:

        raise ValueError("provide close and volume (or a single OHLCV)")

    else:
        close, volume = _as1d(close), _as1d(volume)

    return _obv_kernel(close, volume)


def vwap(high, low=None, close=None, volume=None,
         w: int | None = None) -> NDArray[np.float64]:
    r""" Volume-Weighted Average Price of the typical price, causal.

    Typical price :math:`TP_t = (H_t + L_t + C_t) / 3`, weighted by volume.
    With ``w=None`` it is the **anchored** (cumulative) VWAP; with an integer
    ``w`` it is a rolling VWAP over the trailing window.

    Parameters
    ----------
    high, low, close, volume : array-like, or high = OHLCV
        OHLCV series, or a single :class:`~fynance.core.OHLCV` (volume required).
    w : int, optional
        Rolling window; ``None`` (default) = cumulative.

    Returns
    -------
    numpy.ndarray
        VWAP series, aligned with the input.

    Examples
    --------
    >>> import numpy as np
    >>> h = np.array([10., 12., 11.])
    >>> low = np.array([8., 10., 9.])
    >>> c = np.array([9., 11., 10.])
    >>> v = np.array([100., 100., 100.])
    >>> vwap(h, low, c, v).round(4)
    array([ 9., 10., 10.])

    """
    if isinstance(high, OHLCV):
        bars = high
        high, low, close, volume = bars.high, bars.low, bars.close, bars.volume

    elif low is None or close is None or volume is None:

        raise ValueError(
            "provide high, low, close and volume (or a single OHLCV)"
        )

    else:
        high, low = _as1d(high), _as1d(low)
        close, volume = _as1d(close), _as1d(volume)

    tp = (high + low + close) / 3.0
    win = 0 if w is None else int(w)

    return _vwap_kernel(tp, volume, win)
