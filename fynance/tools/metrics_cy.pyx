#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

from fynance.tools.momentums_cy import smstd_cy

# TODO list:
# - Calmar
# - Rolling calmar
# - move pure python function
# - MDD 
# - Rolling MDD


__all__ = [
    'sharpe_cy', 'roll_sharpe_cy', 'log_sharpe_cy',
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
    
    cdef np.float64_t T = <double>len(series)
    
    if T == 0.:
        return 0.
    
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = series[1:] / series[:-1] - 1.
    cdef np.float64_t annual_return = np.float_power(series[-1] / series[0], <double>period / T, dtype=np.float64) - 1.
    cdef np.float64_t vol = sqrt(<double>period) * np.std(ret_vect, dtype=np.float64)
    
    if vol == 0.:
        return 0.

    return annual_return / vol


cpdef np.ndarray[np.float64_t, ndim=1] roll_sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        np.float64_t period=252,
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
    cdef np.float64_t annual_return = (<double>period / T) * np.sum(ret_vect, dtype=np.float64)
    cdef np.float64_t vol = sqrt(<double>period) * np.std(ret_vect, dtype=np.float64)

    if vol == 0.:
        return 0.
    return annual_return / vol
