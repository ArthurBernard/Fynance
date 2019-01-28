#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from fynance.tools.momentums import *

__all__ = [
    'z_score', 'rsi', 'bollinger_band', 'hma', 'macd_line', 
    'signal_line', 'macd_hist',
]

#=============================================================================#
#                                    Tools                                    #
#=============================================================================#


def z_score(series, kind_ma='ema', **kwargs):
    """ 
    Compute a Z-score function for a specific moving average function. 
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Series of index, prices or returns.
    :kind_ma: str (default 'ema')
        Kind of moving average, eg: 'ema', 'sma' or 'wma'.
    :kwargs: Any parameters for the moving average function.

    Returns
    -------
    :z: np.ndarray[np.float64, ndim=1]
        Z-score at each period.

    Examples
    --------
    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> z_score(series)
    array([ 0.        ,  3.83753384,  1.04129457,  3.27008748,  3.23259291,
           -0.00963602])
    >>> z_score(series, kind_ma='sma', lags=3)
    array([ 0.        ,  1.        , -0.26726124,  1.22474487,  1.22474487,
           -1.22474487])

    """
    if kind_ma.lower() == 'wma':
        ma_f = wma
        std_f = wmstd
    elif kind_ma.lower() == 'sma':
        ma_f = sma
        std_f = smstd
    else:
        ma_f = ema
        std_f = emstd
    m = ma_f(series, **kwargs)
    s = std_f(series, **kwargs)
    s[s == 0.] = 1.
    z = (series - m) / s
    return z


#=============================================================================#
#                                 Indicators                                  #
#=============================================================================#


def rsi(series, kind_ma='ema', lags=21, alpha=None):
    """ 
    Relative Strenght Index is the average gain of upward periods (noted U) 
    divided by the average loss of downward (noted D) periods during the 
    specified time frame, such that : RSI = 100 - 100 / (1 + U / D)

    Paramaters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Index series.
    :kind_ma: str (default is 'ema')
        Kind of maving average, eg: 'ma', 'sma' or 'ema'.
    :lags: int (default is None)
        Number of days.
    :alpha: float (default 0.94)
        Multiplier, 0.94 corresponding at 20 (or 30 ?) lags days.
    
    Returns
    -------
    :RSI: np.ndarray[dtype=np.float64, ndim=1]
        Value of RSI for each period.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> rsi(series, lags=3)
    array([ 0.        , 99.99999804, 69.59769254, 85.55610891, 91.72201613,
           30.00294321])

    """
    series = series.flatten()
    T = np.size(series)
    U = np.zeros([T-1])
    D = np.zeros([T-1])
    delta = np.log(series[1:] / series[:-1])
    U[delta > 0] = delta[delta > 0]
    D[delta < 0] = - delta[delta < 0]
    if kind_ma.lower() == 'ma' or kind_ma.lower() == 'sma':
        ma_U = sma(U, lags=lags)
        ma_D = sma(D, lags=lags)
    elif kind_ma.lower() == 'ema':
        ma_U = ema(U, lags=lags, alpha=alpha)
        ma_D = ema(D, lags=lags, alpha=alpha)
    elif kind_ma.lower() == 'wma':
        ma_U = wma(U, lags=lags)
        ma_D = wma(D, lags=lags)
    else:
        print('Kind moving average is miss specified, \
            exponential moving average is selected by default.')
        ma_U = ema(U, lags=lags, alpha=alpha)
        ma_D = ema(D, lags=lags, alpha=alpha)
    RSI = np.zeros([T])
    RSI[1:] = 100 * ma_U / (ma_U + ma_D + 1e-8)
    return RSI


def bollinger_band(series, lags=21, n_std=2, kind_ma='sma'):
    """ 
    Compute the bollinger bands.

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Series.
    :lags: int (default 21)
        Number of lags for ma.
    :n_std: float (default 1)
        Number of standard deviation.
    :kind_ma: str (default 'sma')
        Kind of moving average, eg: 'sma', 'ema', etc.

    Returns
    -------
    :ma: np.ndarray[dtype=np.float64, ndim=1]
        Moving average of series.
    :n_std * std: np.ndarray[dtype=np.float64, ndim=1]
        Moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> bollinger_band(series, lags=3)
    (array([ 60.,  80.,  80., 100., 120., 120.]), array([ 0.        , 40.        , 32.65986324, 32.65986324, 65.31972647,
           65.31972647]))

    """
    if kind_ma.lower() == 'sma':
        ma = sma(series, lags=lags)
        std = smstd(series, lags=lags)
    elif kind_ma.lower() == 'ema':
        ma = ema(series, lags=lags)
        std = emstd(series, lags=lags)
    elif kind_ma.lower() == 'wma':
        ma = wma(series, lags=lags)
        std = wmstd(series, lags=lags)
    else:
        ma = ema(series, lags=lags)
        std = smstd(series, lags=lags)
    return ma, n_std * std


def hma(series, lags=21, kind_ma='wma'):
    """
    Indicator Hull moving average following:
    hma = wma(2 * wma(x, k / 2) - wma(x, k), sqrt(k))

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Series of prices or returns.
    :lags: int (default 21)
        Number of lags for ma.
    :kind_ma: str (default is 'wma')
        Kind of moving average, eg: 'wma', 'sma' or 'ema'
    
    Returns
    -------
    :hma: np.ndarray[dtype=np.float64, ndim=1]
        Hull moving average of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> hma(series, lags=3)
    array([ 60.        , 113.33333333,  76.66666667, 136.66666667,
           186.66666667,  46.66666667])

    """
    if kind_ma.lower() == 'ema':
        f = ema
    elif kind_ma.lower() == 'sma':
        f = sma
    else:
        f = wma
    wma1 = f(series, lags=int(lags / 2))
    wma2 = f(series, lags=lags)
    hma = f(2 * wma1 - wma2, lags=int(np.sqrt(lags)))
    return hma


def macd_line(series, fast_ma=12, slow_ma=26, kind_ma='ema'):
    """ 
    Indicator Moving Average Convergence Divergence Line:
    
    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Series of index or returns.
    :fast_ma: int (default 12)
        Number of lags for short ma.
    :slow_ma: int (default 26)
        Number of lags for long ma.
    :kind_ma: str (default 'ema')
        Kind of moving average, eg: 'sma', 'ema' or 'wma'.
    
    Returns
    -------
    :macd_lin: np.ndarray[dtype=np.float64, ndim=1]
        Moving avg convergence/divergence line of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> macd_line(series, fast_ma=2, slow_ma=4)
    array([ 0.        , 10.66666722,  4.62222282, 12.84740842, 21.73313755,
           -3.61855386])

    """
    if kind_ma.lower() == 'wma':
        f = wma
    elif kind_ma.lower() == 'sma':
        f = sma
    else:
        f = ema
    fast = f(series, lags=fast_ma)
    slow = f(series, lags=slow_ma)
    macd_lin = fast - slow
    return macd_lin


def signal_line(series, lags=9, fast_ma=12, slow_ma=26, kind_ma='ema'):
    """ 
    Signal Line for k lags with slow and fast lenght 
    
    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Index or returns ?
    :lags: int (default 9)
        Number of lags.
    :fast_ma: int (default 12)
        Number of lags for short ma.
    :slow_ma: int (default 26)
        Number of lags for long ma.
    :kind_ma: str (default 'ema')
        Kind of moving average, eg: 'sma', 'ema' or 'wma'.
    
    Returns
    -------
    :sign_lin: np.ndarray[dtype=np.float64, ndim=1]
        Signal line of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> signal_line(series, lags=3, fast_ma=2, slow_ma=4)
    array([ 0.        ,  5.33333361,  4.97777822,  8.91259332, 15.32286544,
            5.85215579])

    """
    macd_lin = macd_line(series, fast_ma=fast_ma, slow_ma=slow_ma)
    if kind_ma.lower() == 'wma':
        f = wma
    elif kind_ma.lower() == 'sma':
        f = sma
    else:
        f = ema
    sig_lin = f(macd_lin, lags=lags)
    return sig_lin


def macd_hist(series, lags=9, fast_ma=12, slow_ma=26, kind_ma='ema'):
    """ 
    Moving Average Convergence Divergence Histogram 

    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Series of index or returns.
    :lags: int (default 9)
        Number of lags.
    :fast_ma: int (default 12)
        Number of lags for short ma.
    :slow_ma: int (default 26)
        Number of lags for long ma.
    :kind_ma: str (default 'ema')
        Kind of moving average, eg: 'sma', 'ema' or 'wma'.
    
    Returns
    -------
    :hist: np.ndarray[dtype=np.float64, ndim=1]
        Moving avg convergence/divergence histogram of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> macd_hist(series, lags=3, fast_ma=2, slow_ma=4)
    array([ 0.        ,  5.33333361, -0.35555539,  3.9348151 ,  6.41027212,
           -9.47070965])

    """
    macd_lin = macd_line(
        series, fast_ma=fast_ma, slow_ma=slow_ma, kind_ma=kind_ma
    )
    sig_lin = signal_line(
        series, lags=lags, fast_ma=fast_ma, slow_ma=slow_ma, kind_ma=kind_ma
    )
    hist = macd_lin - sig_lin
    return hist

if __name__ == '__main__':
    import doctest
    doctest.testmod()