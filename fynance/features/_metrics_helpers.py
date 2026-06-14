#!/usr/bin/env python3
# coding: utf-8

""" Private helpers shared by the metrics submodules. """

from __future__ import annotations

# Built-in packages
from warnings import warn

# Third-party packages
import numpy as np

# Local packages
from fynance.features.metrics_cy import *
from fynance.features.momentums import _ema, _emstd, _sma, _smstd, _wma, _wmstd

__all__ = ['_annual_return', '_compute_returns', '_annual_volatility', '_annual_downside_volatility', '_drawdown', '_roll_annual_return', '_roll_annual_volatility', '_roll_drawdown', '_roll_mdd', '_roll_annual_return_py', '_roll_annual_volatility_py', '_handler_ma', '_handler_mstd']

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

        return np.asarray(drawdown_cy_2d(X, int(raw)))

    return np.asarray(drawdown_cy_1d(X, int(raw)))


def _roll_annual_return(X, period, w, ddof):
    if ddof >= w:

        raise ValueError(
            'size of the lagged window (w={}) must be strictly greater than '
            'degree of freedom (ddof={})'.format(w, ddof)
        )

    elif (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    elif len(X.shape) == 2:

        return np.asarray(roll_annual_return_cy_2d(X, period, w, ddof))

    return np.asarray(roll_annual_return_cy_1d(X, period, w, ddof))


def _roll_annual_volatility(X, period, log, w, axis, ddof):
    if ddof >= w:

        raise ValueError(
            'size of the lagged window (w={}) must be strictly greater than '
            'degree of freedom (ddof={})'.format(w, ddof)
        )

    elif len(X.shape) == 2:

        return np.asarray(roll_annual_volatility_cy_2d(
            X, period, int(log), w, ddof
        ))

    return np.asarray(roll_annual_volatility_cy_1d(
        X, period, int(log), w, ddof
    ))


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

        return np.asarray(roll_drawdown_cy_2d(X, w, int(raw)))

    return np.asarray(roll_drawdown_cy_1d(X, w, int(raw)))


def _roll_mdd(X, w, raw):
    if len(X.shape) == 2:

        return np.asarray(roll_mdd_cy_2d(X, w, int(raw)))

    return np.asarray(roll_mdd_cy_1d(X, w, int(raw)))


def _roll_annual_return_py(X, period, w, ddof):
    """ Old function. """
    if (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    cum_ret = np.zeros(X.shape)
    cum_ret[: w] = X[: w] / X[0]
    cum_ret[w:] = X[w:] / X[: -w]

    if (cum_ret < 0).any():

        raise ValueError('all values of X must be of the same sign.')

    T = X.shape[0]
    power = period / np.arange(1, T - ddof + 1, dtype=np.float64)

    if len(X.shape) == 2:
        power = power.reshape([T, 1])

    sign = np.sign(X[0])

    anu_ret = np.zeros(X.shape)
    anu_ret[ddof:] = sign * np.float_power(cum_ret[ddof:], power) - 1.

    return anu_ret


def _roll_annual_volatility_py(X, period, log, w, axis, ddof):
    """ Old function. """
    shape = X.shape
    T = shape[0]
    R = np.zeros(shape)
    anu_vol = np.zeros(shape)

    if log:
        R[1:] = np.log(X[1:] / X[:-1])

    else:
        R[1:] = X[1:] / X[:-1] - 1.

    for t in range(ddof + 1, T):
        t0 = max(0, t - w)
        anu_vol[t] = np.std(R[t0:t + 1], axis=axis, ddof=ddof)

    return np.sqrt(period) * anu_vol

