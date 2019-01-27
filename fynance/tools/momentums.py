#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .momentums_cy import *

__all__ = [
    'sma', 'wma', 'ema', 'smstd', 'wmstd', 'emstd',
]

#=============================================================================#
#                               Moving Averages                               #
#=============================================================================#


def sma(series, lags=21):
    """ 
    Simple moving average along k lags. 

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    :lags: int (default 21)
        Number of lags for ma.
    
    Returns
    -------
    :ma: np.ndarray[dtype=np.float64, ndim=1]
        Moving average of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> sma(series, lags=3)
    array([ 60.,  80.,  80., 100., 120., 120.])

    """
    return sma_cy(series.flatten().astype(np.float64), lags=int(lags))


def wma(series, lags=21):
    """ 
    Weighted moving average along k lags. 

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    :lags: int (default 21)
        Number of lags for ma.
    
    Returns
    -------
    :ma: np.ndarray[dtype=np.float64, ndim=1]
        Moving average of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> wma(series, lags=3)
    array([ 60.        ,  86.66666667,  83.33333333, 103.33333333,
           133.33333333, 113.33333333])

    """
    return wma_cy(series.flatten().astype(np.float64), lags=int(lags))


def ema(series, alpha=0.94, lags=None):
    """ 
    Exponential moving average along k lags. 

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Index or returns ?
    :alpha: float (default 0.94)
        Multiplier, 0.94 corresponding at 20 (or 30 ?) lags memory.
    :lags: int (default is None)
        Number of days. If not None => alpha = 1 - 2 / (1 + lags)
    
    Returns
    -------
    :ma: np.ndarray[dtype=np.float64, ndim=1]
        Moving average of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> ema(series, lags=3)
    array([ 60.,  80.,  80., 100., 130., 105.])
    >>> ema(series, alpha=0.5)
    array([ 60.,  80.,  80., 100., 130., 105.])

    """
    if lags is not None:
        alpha = 1 - 2 / (1 + lags)
    return ema_cy(series.flatten().astype(np.float64), alpha=float(alpha))


#=============================================================================#
#                          Moving Standard Deviation                          #
#=============================================================================#


def smstd(series, lags=21):
    """
    Simple moving standard deviation along k lags.

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    :lags: int (default 21)
        Number of lags for ma.
    
    Returns
    -------
    :std: np.ndarray[dtype=np.float64, ndim=1]
        Moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> smstd(series, lags=3)
    array([ 0.        , 20.        , 16.32993162, 16.32993162, 32.65986324,
           32.65986324])

    """
    return smstd_cy(series.flatten().astype(np.float64), lags=int(lags))


def wmstd(series, lags=21):
    """ 
    Weighted moving standard deviation along k lags. 

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Series of prices, index or returns.
    :alpha: float (default 0.94)
        Multiplier, 0.94 corresponding at 20 (or 30 ?) lags memory.
    :lags: int (default is None)
        Number of days. If not None => alpha = 1 - 2 / (1 + lags)
    
    Returns
    -------
    :std: np.ndarray[dtype=np.float64, ndim=1]
        Moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> wmstd(series, lags=3)
    array([ 0.        , 18.85618083, 13.74368542, 17.95054936, 29.8142397 ,
           35.90109871])


    """
    return wmstd_cy(series.flatten().astype(np.float64), lags=int(lags))


def emstd(series, alpha=0.94, lags=None):
    """ 
    Exponential moving standard deviation along k lags. 

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Index or returns ?
    :alpha: float (default 0.94)
        Multiplier, 0.94 corresponding at 20 (or 30 ?) lags memory.
    :lags: int (default is None)
        Number of days. If not None => alpha = 1 - 2 / (1 + lags)
    
    Returns
    -------
    :std: np.ndarray[dtype=np.float64, ndim=1]
        Moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> emstd(series, lags=3)
    array([ 0.        , 28.28427125, 20.        , 31.6227766 , 47.95831523,
           48.98979486])

    """
    if lags is not None:
        alpha = 1 - 2 / (1 + lags)
    return emstd_cy(series.flatten().astype(np.float64), alpha=float(alpha))

if __name__ == '__main__':
    import doctest
    doctest.testmod()