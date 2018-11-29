#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from libc.math cimport sqrt


# TODO list:
# - Rolling sharpe (to check and clean)
# - Calmar
# - Rolling calmar
# - move pure python function


__all__ = [
    'sharpe_cy', 'roll_sharpe', 'roll_sharpe_cy', 'log_sharpe_cy',
]


cpdef np.float64_t sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        np.float64_t period=252.
    ):
    """ 
    Function to compute the total return over the volatility, known as the 
    Sharpe ratio.
    
    Parameters
    ----------
    series: numpy.ndarray(dim=1, dtype=float)
        Prices of the index.
    period: int (default: 252)
        Number of period per year.

    Returns
    -------
    Float, it's the Sharpe ratio.
    """
    if series[0] == 0.:
        return 0. 
    
    cdef np.float64_t T = len(series)
    
    if T == 0.:
        return 0.
    
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = series[1:] / series[:-1] - 1.
    cdef np.float64_t annual_return = np.float_power(series[-1] / series[0], period / T, dtype=np.float64) - 1.
    cdef np.float64_t vol = sqrt(period) * np.std(ret_vect, dtype=np.float64)
    
    if vol == 0.:
        return 0.

    return annual_return / vol


def roll_sharpe(series, period=252, win=0):
    """
    roll_sharpe(series, period=252, win=0)
    
    Vectorized rolling sharpe 
    Parameters
    ----------
    series: np.ndarray[dtype=np.float64, ndim=1]
        Financial series of prices or indexed values.
    period: int (default 252)
        Number of period in a year.
    win: int (default 0)
        Size of the rolling window. If less of two, 
    rolling sharpe is compute on all the past.
    
    Returns
    -------
    rolling_sharpe: np.ndarray[np.float64, ndim=1]
    """
    T = len(series)
    t = np.arange(1, T + 1)
    ret_vect = np.zeros([T])
    ret_vect[1: ] = series[1: ] / series[: -1] - 1
    # Compute rolling cumulative returns
    ret_cum = series / series[0] 
    if win > 1:
        ret_cum[win: ] = series[win: ] / series[: -win]
    # Compute rolling mean
    ret_mean = np.cumsum(ret_vect) / t
    if win > 1:
        ret_mean[win: ] = (ret_mean[win: ] * t[win: ] - ret_mean[: -win] * t[: -win]) / win
    # Compute rolling volatility
    ret_vol = np.cumsum(np.square(ret_vect - ret_mean)) / t
    if win > 1:
        ret_vol[win: ] = (ret_vol[win: ] * t[win: ] - ret_vol[: -win] * t[: -win]) / win
    ret_vol[ret_vol == 0] = 1e10
    # Compute rolling sharpe
    if win > 1:
        t[win: ] = win
    return (np.float_power(ret_cum, period / t) - 1.) / (np.sqrt(period * ret_vol))


cpdef np.ndarray[np.float64_t, ndim=1] roll_sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        np.float64_t period=252,
        int window=0
    ):
    """
    Rolling sharpe
    /! Not optimized /!
    """
    cdef int T = len(series)
    cdef Py_ssize_t t
    cdef np.ndarray[np.float64_t, ndim=1] roll_s = np.zeros([T], dtype=np.float64)
    if window < 2:
        for t in range(1, T):
            roll_s[t] = sharpe_cy(series[:t + 1])
    else:
        for t in range(1, T):
            roll_s[t] = sharpe_cy(series[t - window: t + 1], period=period)
    return roll_s


cpdef np.float64_t log_sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        np.float64_t period=252
    ):
    """ 
    Function to compute the total return over the volatility, known as the 
    Sharpe ratio.
    
    Parameters
    ----------
    series: numpy.ndarray(dim=1, dtype=float)
        Prices of the index.
    period: int (default: 252)
        Number of period per year.

    Returns
    -------
    Float, it's the Sharpe ratio.
    """

    cdef np.float64_t T = series.size 
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = np.log(series[1:] / series[:-1], dtype=np.float64)
    cdef np.float64_t annual_return = (period / T) * np.sum(ret_vect, dtype=np.float64)
    cdef np.float64_t vol = sqrt(period) * np.std(ret_vect, dtype=np.float64)

    if vol == 0.:
        return 0.
    return annual_return / vol
