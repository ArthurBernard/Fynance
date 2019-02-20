#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from fynance.tools.metrics_cy import *
from fynance.tools.momentums_cy import smstd_cy

# TODO:
# - Append window size on rolling calmar
# - Append window size on rolling MDD
# - Append performance 
# - Append rolling performance


__all__ = [
    'sharpe', 'mdd', 'calmar', 'roll_sharpe', 'roll_mdd', 'roll_calmar',
    'drawdown', 'accuracy',
]


#=============================================================================#
#                                   Metrics                                   #
#=============================================================================#


def drawdown(series):
    """ Function to compute measure of the decline from a historical peak in 
    some variable [1]_ (typically the cumulative profit or total open equity 
    of a financial trading strategy). 
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Series of DrawDown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)

    Examples
    --------
    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> drawdown(series)
    array([0. , 0. , 0.2, 0. , 0. , 0.5])

    See Also
    --------
    mdd, calmar, sharpe, roll_mdd

    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return drawdown_cy(series)


def mdd(series):
    """ Function to compute the maximum drwdown where drawdown is the measure 
    of the decline from a historical peak in some variable [2]_ (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.float64
        Scalar of Maximum DrawDown.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Drawdown_(economics)
    
    Examples
    --------
    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> mdd(series)
    0.5

    See Also
    --------
    drawdown, calmar, sharpe, roll_mdd

    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return mdd_cy(series)


def calmar(series, period=252):
    """ Function to compute the compouned annual return over the Maximum 
    DrawDown, known as the Calmar ratio.
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    period : int, optional
        Number of period per year, default is 252 (trading days).

    Returns
    -------
    np.float64
        Scalar of Calmar ratio.

    Examples
    --------
    Assume a series of monthly prices:

    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> calmar(series, period=12)
    0.6122448979591835

    See Also
    --------
    mdd, drawdown, sharpe, roll_calmar

    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return calmar_cy(series, period=float(period))


def sharpe(series, period=252, log=False):
    """ Function to compute the total return over the volatility, known as the 
    Sharpe ratio.
    
    Parameters
    ----------
    series : numpy.ndarray(dim=1, dtype=float)
        Prices of the index.
    period : int, optional
        Number of period per year, default is 252 (trading days).
    log : bool, optional
        If true compute sharpe with the formula for log-returns, default 
        is False.

    Returns
    -------
    np.float64
        Scalar of Sharpe ratio.

    Examples
    --------
    Assume a series of monthly prices:
    
    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> sharpe(series, period=12)
    0.22494843872918127
    
    See Also
    --------
    mdd, calmar, drawdown, roll_sharpe

    """
    series = np.asarray(series, dtype=np.float64).flatten()
    if log:
        return log_sharpe_cy(series, period=float(period))
    return sharpe_cy(series, period=float(period))

# TODO : perf metric and rolling perf metric
#def perf(series, signals=None, exp=False):
#    if signals is None:
#        signals = np.ones(series.shape[0])
#    if exp:
#        f = np.exp
#    else:
#        f = lambda x: x
#    np.cumsum(series * signals) 


#=============================================================================#
#                               Rolling metrics                               #
#=============================================================================#


def roll_mdd(series):
    """ Function to compute the rolling maximum drwdown where drawdown is the 
    measure of the decline from a historical peak in some variable [3]_ 
    (typically the cumulative profit or total open equity of a financial 
    trading strategy).
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    win : int, optional /! NOT YET WORKING /!
        Size of the rolling window. If less of two, rolling Max DrawDown is 
        compute on all the past. Default is 0.

    Returns
    -------
    np.ndrray[np.float64, ndim=1]
        Series of rolling Maximum DrawDown.

    References
    ----------
    .. [3] https://en.wikipedia.org/wiki/Drawdown_(economics)

    Examples
    --------
    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> roll_mdd(series)
    array([0. , 0. , 0.2, 0.2, 0.2, 0.5])
    
    See Also
    --------
    mdd, roll_calmar, roll_sharpe, drawdown

    """
    series = np.asarray(series, dtype=np.float64).flatten()
    return roll_mdd_cy(series)


def roll_calmar(series, period=252.):
    """ Function to compute the rolling compouned annual return over the 
    rolling Maximum DrawDown, that give the rolling Calmar ratio.
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    period : int, optional
        Number of period per year, default is 252 (trading days).
    win : int, optional /! NOT YET WORKING /!
        Size of the rolling window. If less of two, rolling calmar is 
        compute on all the past. Default is 0.

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Series of rolling Calmar ratio.
    
    Examples
    --------
    Assume a monthly series of prices:

    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> roll_calmar(series, period=12)
    array([ 0.        ,  0.        ,  3.52977926, 20.18950437, 31.35989887,
            0.6122449 ])

    See Also
    --------
    roll_mdd, roll_sharpe, calmar

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


def roll_sharpe(series, period=252, win=0, cap=True):
    """ Vectorized function to compute rolling sharpe (compouned annual 
    returns divided by annual volatility).
    
    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Financial series of prices or indexed values.
    period : int, optional
        Number of period in a year, default is 252 (trading days).
    win : int, optional
        Size of the rolling window. If less of two, rolling sharpe is 
        compute on all the past. Default is 0.
    cap : bool, optional
        Cap extram values (some time due to small size window), default 
        is True.
    
    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Serires of rolling Sharpe ratio.
    
    Examples
    --------
    Assume a monthly series of prices:

    >>> series = np.array([70, 100, 80, 120, 160, 80])
    >>> roll_sharpe(series, period=12)
    array([0.        , 0.        , 0.77721579, 3.99243019, 6.754557  ,
           0.24475518])

    See Also
    --------
    roll_calmar, sharpe, roll_mdd

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
        if win == T:
            win = T // 3
        s = np.std(roll_shar[win:])
        m = np.mean(roll_shar[win:])
        xtrem_val = np.abs(roll_shar[:win]) > s * m
        roll_shar[:win][xtrem_val] = 0.
    
    return roll_shar


def accuracy(y_true, y_pred, sign=True):
    """ Compute the accuracy of prediction. 
    
    Parameters
    ----------
    y_true : np.ndarray[ndim=1, dtype=np.float64]
        Vector of true series.
    y_pred : np.ndarray[ndim=1, dtype=np.float64]
        Vector of predicted series.
    sign : bool, optional
        Check sign accuracy if true, else check exact accuracy, default 
        is True.
    
    Returns
    -------
    float
        Accuracy of prediction as float between 0 and 1.
    
    Examples
    --------
    >>> y_true = np.array([1., .5, -.5, .8, -.2])
    >>> y_pred = np.array([.5, .2, -.5, .1, .0])
    >>> accuracy(y_true, y_pred)
    0.8
    >>> accuracy(y_true, y_pred, sign=False)
    0.2

    See Also
    --------
    mdd, calmar, sharpe, drawdown

    """
    if sign:
        y_true = np.sign(y_true)
        y_pred = np.sign(y_pred)
    # Check right answeres
    R = np.sum(y_true == y_pred)
    # Check wrong answeres
    W = np.sum(y_true != y_pred)
    return R / (R + W)

if __name__ == '__main__':
    import doctest
    doctest.testmod()