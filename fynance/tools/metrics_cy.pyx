#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

from fynance.tools.momentums_cy import smstd_cy

# TODO list:
# - Append window size on rolling calmar
# - Append window size on rolling MDD


__all__ = [
    'sharpe_cy', 'roll_sharpe_cy', 'log_sharpe_cy', 'mdd_cy', 'calmar_cy', 
    'roll_mdd_cy', 'roll_calmar_cy',
]


#=============================================================================#
#                                   Metrics                                   #
#=============================================================================#


cpdef np.float64_t mdd_cy(np.ndarray[np.float64_t, ndim=1] series):
    """
    Function to compute the maximum drwdown where drawdown is the measure of 
    the decline from a historical peak in some variable (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    :out: np.float64
        Maximum DrawDown.

    Note
    ----
    Source: https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    cdef np.ndarray[np.float64_t, ndim=1] maximums, drawdowns
    
    maximums = np.maximum.accumulate(series, dtype=np.float64)
    drawdowns = 1. - series / maximums

    return np.max(drawdowns, dtype=np.float64)


cpdef np.float64_t calmar_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        np.float64_t period=252.
    ):
    """
    Function to compute the compouned annual return over the Maximum DrawDown, 
    known as the Calmar ratio.
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    :period: int (default 252)
        Number of period per year.

    Returns
    -------
    :out: np.float64
        Calmar ratio.
    """
    cdef np.float64_t ret, annual_return, max_dd, T = series.size
    
    ret = series[-1] / series[0]
    annual_return = np.float_power(ret, period / T, dtype=np.float64) - 1.
    max_dd = mdd_cy(series)
    if max_dd == 0.:
        return 0.

    return annual_return / max_dd


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
    
    cdef np.float64_t T = <double>len(series)
    
    if T == 0.:
        return 0.
    
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = series[1:] / series[:-1] - 1.
    cdef np.float64_t annual_return = np.float_power(series[-1] / series[0], <double>period / T, dtype=np.float64) - 1.
    cdef np.float64_t vol = sqrt(<double>period) * np.std(ret_vect, dtype=np.float64)
    
    if vol == 0.:
        return 0.

    return annual_return / vol


cpdef np.float64_t log_sharpe_cy(
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

    cdef np.float64_t T = series.size 
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = np.log(series[1:] / series[:-1], dtype=np.float64)
    cdef np.float64_t annual_return = (<double>period / T) * np.sum(ret_vect, dtype=np.float64)
    cdef np.float64_t vol = sqrt(<double>period) * np.std(ret_vect, dtype=np.float64)

    if vol == 0.:
        return 0.
    return annual_return / vol


#=============================================================================#
#                               Rolling metrics                               #
#=============================================================================#


cpdef np.ndarray[np.float64_t, ndim=1] roll_mdd_cy(
        np.ndarray[np.float64_t, ndim=1] series
    ):
    """
    Function to compute the maximum drwdown where drawdown is the measure of 
    the decline from a historical peak in some variable (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    :out: np.ndarray[np.float64, ndim=1]
        Maximum DrawDown.

    Note
    ----
    Source: https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    cdef np.ndarray[np.float64_t, ndim=1] maximums, drawdowns
    
    maximums = np.maximum.accumulate(series, dtype=np.float64)
    drawdowns = 1. - series / maximums

    return np.maximum.accumulate(drawdowns, dtype=np.float64)


cpdef np.ndarray[np.float64_t, ndim=1] roll_calmar_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        np.float64_t period=252.
    ):
    """
    Function to compute the rolling compouned annual return over the rolling 
    Maximum DrawDown, known as the rolling Calmar ratio.
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    :period: int (default 252)
        Number of period per year.

    Returns
    -------
    :out: np.ndarray[np.float64, ndim=1]
        Calmar ratio.
    """
    cdef np.float64_t T = series.size
    cdef np.ndarray[np.float64_t, ndim=1] t, ret, annual_return

    ret = np.zeros([<int>T], dtype=np.float64)
    t = np.arange(1., T + 1., dtype=np.float64)
    ret[1:] = series[1:] / series[0]
    annual_return = np.float_power(ret, period / t, dtype=np.float64) - 1.
    
    return annual_return / (roll_mdd_cy(series) + 1e-8)


cpdef np.ndarray[np.float64_t, ndim=1] roll_sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        np.float64_t period=252.,
        int window=0
    ):
    """
    Rolling sharpe
    /! Not optimized /!
    """
    cdef int t, T = len(series)
    cdef np.ndarray[np.float64_t, ndim=1] roll_s = np.zeros([T], dtype=np.float64)
    
    for t in range(1, T):
        if window > t or window < 2:
            roll_s[t] = sharpe_cy(series[:t + 1], period=period)
        else:
            roll_s[t] = sharpe_cy(series[t - window: t + 1], period=period)
    return roll_s