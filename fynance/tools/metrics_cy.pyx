#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

from fynance.tools.momentums_cy import smstd_cy

# TODO list:
# - Append window size on rolling MDD


__all__ = [
    'sharpe_cy', 'roll_sharpe_cy', 'log_sharpe_cy', 'mdd_cy', 'calmar_cy', 
    'roll_mdd_cy', 'drawdown_cy',
]


#=============================================================================#
#                                   Metrics                                   #
#=============================================================================#


cpdef np.ndarray[np.float64_t, ndim=1] drawdown_cy(
        np.ndarray[np.float64_t, ndim=1] series
    ):
    """
    Function to compute measure of the decline from a historical peak in some 
    variable (typically the cumulative profit or total open equity of a 
    financial trading strategy). 
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    :out: np.ndarray[np.float64, ndim=1]
        DrawDown.

    Note
    ----
    Source: https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    cdef np.ndarray[np.float64_t, ndim=1] maximums
    
    # Compute DrawDown
    maximums = np.maximum.accumulate(series, dtype=np.float64)
    
    return 1. - series / maximums


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
    cdef np.ndarray[np.float64_t, ndim=1] drawdowns
    
    # Compute DrawDown
    drawdowns = drawdown_cy(series)

    return max(drawdowns)


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
    
    # Compute compouned annual returns
    ret = series[-1] / series[0]
    annual_return = np.sign(ret) * np.float_power(
        np.abs(ret), period / T, dtype=np.float64) - 1.
    
    # Compute MaxDrawDown
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
    
    # Compute compouned annual returns
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = series[1:] / series[:-1] - 1.
    cdef np.float64_t annual_return 
    annual_return = np.sign(series[-1] / series[0]) * np.float_power(
        np.abs(series[-1] / series[0]), <double>period / T, dtype=np.float64) - 1.
    
    # Compute annual volatility
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

    # Compute compouned annual returns
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = np.log(series[1:] / series[:-1], dtype=np.float64)
    cdef np.float64_t annual_return = (<double>period / T) * np.sum(ret_vect, dtype=np.float64)
    
    # Compute annual volatility
    cdef np.float64_t vol = sqrt(<double>period) * np.std(ret_vect, dtype=np.float64)

    if vol == 0.:
        return 0.

    return annual_return / vol


#=============================================================================#
#                               Rolling metrics                               #
#=============================================================================#


cpdef np.ndarray[np.float64_t, ndim=1] roll_mdd_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        int win=0
    ):
    """
    Function to compute the maximum drwdown where drawdown is the measure of 
    the decline from a historical peak in some variable (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    :win: int (default 0)
        Size of the rolling window. If less of two, 
        rolling Max DrawDown is compute on all the past.

    Returns
    -------
    :out: np.ndarray[np.float64, ndim=1]
        Series of rolling Maximum DrawDown.

    Note
    ----
    Source: https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    cdef np.ndarray[np.float64_t, ndim=1] drawdowns
    
    # Compute DrawDown
    drawdowns = drawdown_cy(series)
    
    if win <= 2:
        return np.maximum.accumulate(drawdowns, dtype=np.float64)


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