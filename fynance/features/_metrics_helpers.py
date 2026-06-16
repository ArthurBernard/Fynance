#!/usr/bin/env python3
# coding: utf-8

""" Private helpers shared by the metrics submodules. """

from __future__ import annotations

# Built-in packages
from warnings import warn

# Third-party packages
import numpy as np

# Local packages
from numba import njit, prange

# Local packages
from fynance.features.momentums import _ema, _emstd, _sma, _sma_1d, _smstd, _wma, _wmstd

# --------------------------------------------------------------------------- #
#   numba metric kernels (ported 1:1 from the former Cython metrics_cy)        #
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _drawdown_1d(X, raw):
    T = X.shape[0]
    dd = np.empty(T, dtype=np.float64)
    S = X[0]
    for t in range(T):
        if X[t] > S:
            S = X[t]
        if raw != 0:
            dd[t] = S - X[t]
        else:
            dd[t] = 1.0 - X[t] / S
    return dd


@njit(cache=True)
def _drawdown_2d(X, raw):
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        out[:, n] = _drawdown_1d(X[:, n].copy(), raw)
    return out


@njit(cache=True)
def _roll_drawdown_1d(X, w, raw):
    T = X.shape[0]
    if w >= T:
        return _drawdown_1d(X, raw)
    dd = np.empty(T, dtype=np.float64)
    S = X[0]
    for t in range(T):
        if t < w:
            if X[t] > S:
                S = X[t]
        else:
            S = X[t]
            i = 1
            while i < w:
                if X[t - i] > S:
                    S = X[t - i]
                i += 1
        if raw != 0:
            dd[t] = S - X[t]
        else:
            dd[t] = 1.0 - X[t] / S
    return dd


@njit(cache=True)
def _roll_drawdown_2d(X, w, raw):
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        out[:, n] = _roll_drawdown_1d(X[:, n].copy(), w, raw)
    return out


@njit(cache=True)
def _roll_mdd_1d(X, w, raw):
    # First w points: expanding drawdown, running max of it (parity reference).
    T = X.shape[0]
    mdd = np.empty(T, dtype=np.float64)
    S = 0.0
    run = X[0]
    for t in range(min(w, T)):
        if X[t] > run:
            run = X[t]
        dd = (run - X[t]) if raw != 0 else (1.0 - X[t] / run)
        if dd > S:
            S = dd
        mdd[t] = S
    # Trailing windows: max drawdown within X[t-w+1 : t+1], computed in place
    # (no per-window allocation, same arithmetic as the former Cython).
    for t in range(w, T):
        run = X[t - w + 1]
        S = 0.0
        for j in range(t - w + 1, t + 1):
            if X[j] > run:
                run = X[j]
            dd = (run - X[j]) if raw != 0 else (1.0 - X[j] / run)
            if dd > S:
                S = dd
        mdd[t] = S
    return mdd


@njit(parallel=True, cache=True)
def _roll_mdd_2d(X, w, raw):
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in prange(N):
        out[:, n] = _roll_mdd_1d(np.ascontiguousarray(X[:, n]), w, raw)
    return out


@njit(cache=True)
def _roll_annual_return_1d(X, p, w, d):
    T = X.shape[0]
    ar = np.empty(T, dtype=np.float64)
    R = 0.0
    _w = 1.0
    for t in range(T):
        if t < w:
            R = X[t] / X[0]
            _w = float(t + 1 - d)
        else:
            R = X[t] / X[t - w + 1]
        if t < d:
            ar[t] = 0.0
        else:
            ar[t] = R ** (p / _w) - 1.0
    return ar


@njit(cache=True)
def _roll_annual_return_2d(X, p, w, d):
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        out[:, n] = _roll_annual_return_1d(X[:, n].copy(), p, w, d)
    return out


@njit(cache=True)
def _roll_annual_volatility_1d(X, p, ll, w, d):
    T = X.shape[0]
    av = np.empty(T, dtype=np.float64)
    R = np.zeros(T, dtype=np.float64)
    S = 0.0
    S2 = 0.0
    _w = 1.0
    _w_d = 1.0
    sub_R = 0.0
    av[0] = 0.0
    for t in range(1, T):
        if ll != 0:
            R[t] = np.log(X[t] / X[t - 1])
        else:
            R[t] = X[t] / X[t - 1] - 1.0
        if t < w:
            _w = float(t + 1)
            _w_d = float(t + 1 - d)
            sub_R = 0.0
        elif t > w:
            sub_R = R[t - w]
        S += R[t] - sub_R
        S2 += R[t] * R[t] - sub_R * sub_R
        if t < d:
            av[t] = 0.0
        else:
            av[t] = np.sqrt(p * (S2 - (S / _w) * S) / _w_d)
    return av


@njit(cache=True)
def _roll_annual_volatility_2d(X, p, ll, w, d):
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        out[:, n] = _roll_annual_volatility_1d(X[:, n].copy(), p, ll, w, d)
    return out


@njit(cache=True)
def _roll_mad_1d(X, w):
    ma = _sma_1d(X, w)
    T = X.shape[0]
    mad = np.empty(T, dtype=np.float64)
    for t in range(T):
        S = 0.0
        i = 0
        if t < w:
            while i <= t:
                S += abs(X[i] - ma[t])
                i += 1
            mad[t] = S / (t + 1)
        else:
            while i < w:
                S += abs(X[t - i] - ma[t])
                i += 1
            mad[t] = S / w
    return mad


@njit(cache=True)
def _roll_mad_2d(X, w):
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        out[:, n] = _roll_mad_1d(X[:, n].copy(), w)
    return out

__all__ = ['_annual_return', '_compute_returns', '_annual_volatility', '_annual_downside_volatility', '_drawdown', '_roll_annual_return', '_roll_annual_volatility', '_roll_drawdown', '_roll_mdd', '_handler_ma', '_handler_mstd', '_roll_mad_1d', '_roll_mad_2d']

_handler_ma = {'s': _sma, 'w': _wma, 'e': _ema}
_handler_mstd = {'s': _smstd, 'w': _wmstd, 'e': _emstd}


def _annual_return(X, period, ddof):
    if (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    ret = X[-1] / X[0]
    T = X.shape[0]

    if (ret < 0).any():

        raise ValueError('initial value X[0] and final value X[-1] must '
                         'be of the same sign.')

    sign = np.sign(X[0])
    power = period / (T - ddof)

    return sign * np.float_power(ret, power, dtype=np.float64) - 1.


def _compute_returns(X, log):
    R = np.zeros(X.shape)
    if log:
        R[1:] = np.log(X[1:] / X[:-1])
    else:
        R[1:] = X[1:] / X[:-1] - 1.
    return R


def _annual_volatility(X, period, log, axis, ddof):
    return np.sqrt(period) * np.std(_compute_returns(X, log), axis=axis, ddof=ddof)


def _annual_downside_volatility(X, period, log, axis, ddof):
    R = _compute_returns(X, log)
    return np.sqrt(period) * np.std(np.where(R < 0, R, 0.), axis=axis, ddof=ddof)


def _drawdown(X, raw):
    if (X[0] == 0).any() and not raw:

        warn(
            'Cannot compute drawdown in percentage without initial values '
            'X[0] strictly positive.',
            category=UserWarning,
            stacklevel=2,
        )
        raw = True

    if len(X.shape) == 2:

        return _drawdown_2d(X, int(raw))

    return _drawdown_1d(X, int(raw))


def _roll_annual_return(X, period, w, ddof):
    if ddof >= w:

        raise ValueError(
            'size of the lagged window (w={}) must be strictly greater than '
            'degree of freedom (ddof={})'.format(w, ddof)
        )

    elif (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    elif len(X.shape) == 2:

        return _roll_annual_return_2d(X, period, w, ddof)

    return _roll_annual_return_1d(X, period, w, ddof)


def _roll_annual_volatility(X, period, log, w, axis, ddof):
    if ddof >= w:

        raise ValueError(
            'size of the lagged window (w={}) must be strictly greater than '
            'degree of freedom (ddof={})'.format(w, ddof)
        )

    elif len(X.shape) == 2:

        return _roll_annual_volatility_2d(X, period, int(log), w, ddof)

    return _roll_annual_volatility_1d(X, period, int(log), w, ddof)


def _roll_drawdown(X, w, raw):
    if (X[0] == 0).any() and not raw:

        warn(
            'Cannot compute drawdown in percentage without initial values '
            'X[0] strictly positive.',
            category=UserWarning,
            stacklevel=2,
        )
        raw = True

    if len(X.shape) == 2:

        return _roll_drawdown_2d(X, w, int(raw))

    return _roll_drawdown_1d(X, w, int(raw))


def _roll_mdd(X, w, raw):
    if len(X.shape) == 2:

        return _roll_mdd_2d(X, w, int(raw))

    return _roll_mdd_1d(X, w, int(raw))
