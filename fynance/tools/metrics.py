#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from fynance.tools.metrics_cy import *
from fynance.tools.momentums_cy import smstd_cy

# TODO list:
# - Append window size on rolling calmar
# - Append window size on rolling MDD


__all__ = [
    'sharpe', 'mdd', 'calmar', 'roll_sharpe', 'roll_mdd', 'roll_calmar',
    'drawdown',
]


#=============================================================================#
#                                   Metrics                                   #
#=============================================================================#


def drawdown(series):
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
        Series of DrawDown.

    Note
    ----
    Source: https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return drawdown_cy(series)


def mdd(series):
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
        Scalar of Maximum DrawDown.

    Note
    ----
    Source: https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return mdd_cy(series)


def calmar(series, period=252):
    """
    Function to compute the compouned annual return over the Maximum DrawDown, 
    known as the Calmar ratio.
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    :period: int (default 252)
        Number of period per year.
    :log: bool (default False)
        If true compute sharpe with the formula for log-returns

    Returns
    -------
    :out: np.float64
        Scalar of Calmar ratio.
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return calmar_cy(series, period=float(period))


def sharpe(series, period=252, log=False):
    """ 
    Function to compute the total return over the volatility, known as the 
    Sharpe ratio.
    
    Parameters
    ----------
    :series: numpy.ndarray(dim=1, dtype=float)
        Prices of the index.
    :period: int (default: 252)
        Number of period per year.
    :log: bool (default False)
        If true compute sharpe with the formula for log-returns

    Returns
    -------
    :out: np.float64
        Scalar of Sharpe ratio.
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    if log:
        return log_sharpe_cy(series, period=float(period))
    return sharpe_cy(series, period=float(period))


#=============================================================================#
#                               Rolling metrics                               #
#=============================================================================#


def roll_mdd(series):
    """
    Function to compute the rolling maximum drwdown where drawdown is the 
    measure of the decline from a historical peak in some variable (typically 
    the cumulative profit or total open equity of a financial trading 
    strategy).
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    :win: int (default 0) /! NOT YET WORKING /!
        Size of the rolling window. If less of two, 
        rolling Max DrawDown is compute on all the past.

    Returns
    -------
    :out: np.ndrray[np.float64, ndim=1]
        Series of rolling Maximum DrawDown.

    Note
    ----
    Source: https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return roll_mdd_cy(series)


def roll_calmar(series, period=252.):
    """
    Function to compute the rolling compouned annual return over the rolling 
    Maximum DrawDown, that give the rolling Calmar ratio.
    
    Parameters
    ----------
    :series: np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    :period: int (default 252)
        Number of period per year.
    :win: int (default 0) /! NOT YET WORKING /!
        Size of the rolling window. If less of two, 
        rolling calmar is compute on all the past.

    Returns
    -------
    :out: np.ndarray[np.float64, ndim=1]
        Series of rolling Calmar ratio.
    """
    # Set variables
    series = np.asarray(series, dtype=np.float64).flatten()
    T = series.size
    t = np.arange(1., T + 1., dtype=np.float64)
    
    # Compute roll Returns
    ret = series / series[0]
    annual_return = np.sign(ret) * np.float_power(
        np.abs(ret), period / t, dtype=np.float64) - 1.
    
    # Compute roll MaxDrawDown
    roll_maxdd = roll_mdd_cy(series)
    
    # Compute roll calmar
    roll_cal = np.zeros([T])
    not_null = roll_maxdd != 0.
    roll_cal[not_null] = annual_return[not_null] / roll_maxdd[not_null]

    return roll_cal

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
            period=float(period)
        )
    
    # Set variables
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


def roll_sharpe(series, period=252, win=0, cap=True):
    """
    Vectorized function to compute rolling sharpe (compouned annual returns 
    divided by annual volatility).
    
    Parameters
    ----------
    :series: np.ndarray[dtype=np.float64, ndim=1]
        Financial series of prices or indexed values.
    :period: int (default 252)
        Number of period in a year.
    :win: int (default 0)
        Size of the rolling window. If less of two, 
        rolling sharpe is compute on all the past.
    :cap: bool (default True)
        Cap extram values (some time due to small size window).
    
    Returns
    -------
    :out: np.ndarray[np.float64, ndim=1]
        Serires of rolling Sharpe ratio.
    """
    
    # Setting inputs
    series = np.asarray(series, dtype=np.float64).flatten()
    T = series.size
    t = np.arange(1., T + 1., dtype=np.float64)
    if win < 2:
        win = T
    t[t > win] = win + 1.
    ret = np.zeros([T], dtype=np.float64)
    ret[1:] = series[1:] / series[:-1] - 1.
    
    # Compute rolling perf
    ma = series / series[0] 
    ma[win:] = series[win: ] / series[: -win]
    annual_return = np.sign(ma) * np.float_power(
        np.abs(ma), period / t, dtype=np.float64) - 1.
    
    # Compute rolling volatility
    std = smstd_cy(np.asarray(ret).flatten(), lags=int(win))
    vol = np.sqrt(period) * std
    
    # Compute sharpe
    roll_shar = np.zeros([T])
    not_null = vol != 0.
    roll_shar[not_null] = annual_return[not_null] / vol[not_null]

    # Cap extrem value
    if cap:
        xtrem_val = roll_shar[:win] > 10 * np.mean(roll_shar)
        roll_shar[:win][xtrem_val] = 0.
    
    return roll_shar