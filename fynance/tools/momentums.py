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
    """
    return smstd_cy(series.flatten().astype(np.float64), lags=int(lags))


def wmstd(series, lags=21):
    """ 
    Weighted moving standard deviation along k lags. 

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
    """
    if lags is not None:
        alpha = 1 - 2 / (1 + lags)
    return emstd_cy(series.flatten().astype(np.float64), alpha=float(alpha))