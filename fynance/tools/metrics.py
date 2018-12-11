#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from fynance.tools.metrics_cy import sharpe_cy, log_sharpe_cy
from fynance.tools.momentums_cy import smstd_cy


__all__ = [
    'sharpe', 'roll_sharpe',
]


def sharpe(series, period=252, log=False):
    """ 
    Function to compute the total return over the volatility, known as the 
    Sharpe ratio.
    
    Parameters
    ----------
    series: numpy.ndarray(dim=1, dtype=float)
        Prices of the index.
    period: int (default: 252)
        Number of period per year.
    log: bool (default False)
        If true compute sharpe with the formula for log-returns

    Returns
    -------
    Float, it's the Sharpe ratio.
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    if log:
        return log_sharpe_cy(series, period=float(period))
    return sharpe_cy(series, period=float(period))


def roll_sharpe_NOT(series, period=252, win=0):
    """
    /! WRONG FORMULA /!
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
    
    if win < 2:
        return sharpe_cy(
            np.asarray(series, dtype=np.float64).flatten(), 
            period=period
        )

    T = series.size
    t = np.arange(1, T + 1)
    ret_vect = np.zeros([T])
    ret_vect[1: ] = series[1: ] / series[: -1] - 1
    # Compute rolling cumulative returns
    ret_cum = series / series[0] 
    ret_cum[win: ] = series[win: ] / series[: -win]
    # Compute rolling mean
    ret_mean = np.cumsum(ret_vect) / t
    ret_mean[win: ] = (ret_mean[win: ] * t[win: ] - ret_mean[: -win] * t[: -win]) / win
    # Compute rolling volatility
    ret_vol = np.cumsum(np.square(ret_vect - ret_mean)) / t
    ret_vol[win: ] = (ret_vol[win: ] * t[win: ] - ret_vol[: -win] * t[: -win]) / win
    ret_vol[ret_vol == 0] = 1e-8
    # Compute rolling sharpe
    t[win: ] = win
    return (np.float_power(ret_cum, period / t) - 1.) / np.sqrt(period * ret_vol)


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
    # Setting inputs
    series = np.asarray(series, dtype=np.float64).flatten()
    T = series.size
    t = np.arange(1., T + 1., dtype=np.float64)
    t[t > win] = win + 1.
    if win < 2:
        win = T
    ret = series[1:] / series[:-1] - 1.
    # Compute rolling perf
    ma = series / series[0] 
    ma[win:] = series[win: ] / series[: -win]
    annual_return = np.float_power(ma, period / t, dtype=np.float64) - 1.
    # Compute rolling volatility
    std = np.zeros([T])
    std[1:] = smstd_cy(ret, lags=int(win))
    std[std == 0.] = 1e-8
    vol = np.sqrt(period) * std
    return annual_return / vol