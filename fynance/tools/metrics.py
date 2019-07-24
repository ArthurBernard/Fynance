#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2018-12-14 19:11:40
# @Last modified by: ArthurBernard
# @Last modified time: 2019-07-24 17:01:34

""" Metric functons used in financial analysis. """

# Built-in packages

# External packages
import numpy as np

# Internal packages
from fynance.tools.metrics_cy import calmar_cy, drawdown_cy, mdd_cy, sharpe_cy
from fynance.tools.metrics_cy import log_sharpe_cy, roll_mdd_cy, roll_mad_cy
from fynance.tools.momentums_cy import smstd_cy

# TODO:
# - Append window size on rolling calmar
# - Append window size on rolling MDD
# - Append performance
# - Append rolling performance
# - verify and fix error to perf_strat, perf_returns, perf_index


__all__ = [
    'accuracy', 'calmar', 'diversified_ratio', 'drawdown', 'mad', 'mdd',
    'roll_calmar', 'roll_mad', 'roll_mdd', 'roll_sharpe', 'sharpe',
]


# =========================================================================== #
#                                   Metrics                                   #
# =========================================================================== #


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


def calmar(series, period=252):
    """ Compute the compouned annual return over the Maximum DrawDown.

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    period : int, optional
        Number of period per year, default is 252 (trading days).

    Returns
    -------
    np.float64
        Value of Calmar ratio.

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


def diversified_ratio(series, w=None, std_method='std'):
    r""" Compute diversification ratio of a portfolio.

    Diversification ratio, denoted D, is defined as the ratio of the
    portfolio's weighted average volatility to its overll volatility,
    developed by Choueifaty and Coignard [1]_.

    .. math::

        D(P) = \frac{P' \Sigma}{\sqrt{P'VP}}

    With :math:`\Sigma` vector of asset volatilities, :math:`P` vector of
    weights of asset of portfolio, and :math:`V` matrix of variance-covariance
    of these assets.

    Parameters
    ----------
    series : np.array[ndim=2, dtype=np.float64] of shape (T, N)
        Portfolio matrix of N assets and T time periods, each column
        correspond to one series of prices.
    w : np.array[ndim=1 or 2, dtype=np.float64] of size N, optional
        Vector of weights, default is None it means it will equaly weighted.
    std_method : str, optional /!\ Not yet implemented /!\
        Method to compute variance vector and covariance matrix.

    Returns
    -------
    np.float64
        Value of diversification ratio of the portfolio.

    References
    ----------
    .. [1] tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf

    """
    T, N = series.shape

    if w is None:
        w = np.ones([N, 1]) / N
    else:
        w = w.reshape([N, 1])

    sigma = np.std(series, axis=0).reshape([N, 1])
    V = np.cov(series, rowvar=False, bias=True).reshape([N, N])

    return (w.T @ sigma) / np.sqrt(w.T @ V @ w)


def drawdown(series):
    """ Measures the drawdown of `series`.

    Function to compute measure of the decline from a historical peak in some
    variable [1]_ (typically the cumulative profit or total open equity of a
    financial trading strategy).

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


def mad(series):
    """ Compute the Mean Absolute Deviation.

    Compute the mean of the absolute value of the distance to the mean. _[1]

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.float64
        Value of mean absolute deviation.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Average_absolute_deviation

    Examples
    --------
    >>> series = np.array([70., 100., 90., 110., 150., 80.])
    >>> mad(series)
    20.0

    See Also
    --------
    roll_mad

    """
    return np.mean(np.abs(series - np.mean(series)))


def mdd(series):
    """ Compute the maximum drwdown.

    Drawdown is the measure of the decline from a historical peak in some
    variable [2]_ (typically the cumulative profit or total open equity of a
    financial trading strategy).

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.float64
        Value of Maximum DrawDown.

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


def perf_index(series, init_value=100.):
    """ Compute performance of prices or index values along time axis.

    Parameters
    ----------
    series : np.ndarray[ndim=1, dtype=np.float64]
        Time-series of prices or index values.
    init_value : float, optional
        Initial value for measure the performance, default is 100.

    Returns
    -------
    np.ndarray[ndim=1, dtype=np.float64]
        Performances along time axis.

    See Also
    --------
    perf_returns, perf_strat

    Examples
    --------
    >>> series = np.array([10., 12., 15., 14., 16., 18., 16.])
    >>> perf_index(series, init_value=100.)
    array([100., 120., 150., 140., 160., 180., 160.])

    """
    return init_value * series / series[0]


def perf_returns(returns, log=False, init_value=100.):
    """ Compute performance of returns along time axis.

    Parameters
    ----------
    returns : np.ndarray[ndim=1, dtype=np.float64]
        Time-series of returns.
    log : bool, optional
        Considers returns as log-returns if True. Default is False.
    init_value : float, optional
        Initial value for measure the performance, default is 100.

    Returns
    -------
    np.ndarray[ndim=1, dtype=np.float64]
        Performances along time axis.

    See Also
    --------
    perf_index, perf_strat

    Examples
    --------
    >>> returns = np.array([0., 20., 30., -10., 20., 20., -20.])
    >>> perf_returns(returns, init_value=100., log=False)
    array([100., 120., 150., 140., 160., 180., 160.])

    """
    series = np.cumsum(returns) + init_value

    if log:
        series = np.exp(series)

    return perf_index(series, init_value=init_value)


# TODO : finish perf strat metric (add reinvest option)
def perf_strat(underlying, signals=None, log=False, init_value=100.,
               reinvest=False):
    """ Compute the performance of a strategy.

    With respect to this underlying and signal series along time axis.

    Parameters
    ----------
    underlying : np.ndarray[ndim=1, dtype=np.float64]
        Time-series of prices or index values.
    signals : np.ndarray[ndim=1, dtype=np.float64]
        Time-series of signals, if `None` considering a long position.
    log : bool, optional
        Considers underlying series as log values if True. Default is False.
    init_value : float, optional
        Initial value for measure the performance, default is 100.
    reinvest : bool, optional
        Reinvest profit/loss if true.

    Returns
    -------
    np.ndarray[ndim=1, dtype=np.float64]
        Performances along time axis.

    See Also
    --------
    perf_returns, perf_index

    Examples
    --------
    >>> underlying = np.array([10., 12., 15., 14., 16., 18., 16.])
    >>> signals = np.array([1., 1., 1., 0., 1., 1., -1.])
    >>> perf_strat(underlying, signals, init_value=100.)
    array([100., 120., 150., 150., 170., 190., 210.])

    # >>> perf_strat(underlying, signals, init_value=100., reinvest=True)
    # array([100., 120., ])

    """
    returns = np.zeros(underlying.shape)
    underlying *= init_value / underlying[0]
    returns[1:] = underlying[1:] - underlying[:-1]

    if signals is None:
        signals = np.ones(underlying.shape[0])

    series = returns * signals

    return perf_returns(series, log=log, init_value=init_value)


def sharpe(series, period=252, log=False):
    """ Compute the total return over the volatility.

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
        Value of Sharpe ratio.

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


# =========================================================================== #
#                               Rolling metrics                               #
# =========================================================================== #

# TODO : rolling perf metric
# TODO : rolling diversified ratio

def roll_calmar(series, period=252.):
    """ Compute the rolling Calmar ratio.

    It is the compouned annual return over the rolling Maximum DrawDown.

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


def roll_mad(series, win=0):
    """ Compute rolling Mean Absolut Deviation.

    Compute the moving average of the absolute value of the distance to the
    moving average _[1].

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Series of mean absolute deviation.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Average_absolute_deviation

    Examples
    --------
    >>> series = np.array([70., 100., 90., 110., 150., 80.])
    >>> roll_mad(series)
    array([ 0.        , 15.        , 11.11111111, 12.5       , 20.8       ,
           20.        ])

    See Also
    --------
    mad

    """
    series = np.asarray(series, dtype=np.float64).flatten()

    if win < 2:
        win = series.size

    return roll_mad_cy(series, win=int(win))


def roll_mdd(series):
    """ Compute the rolling maximum drwdown.

    Where drawdown is the measure of the decline from a historical peak in
    some variable [3]_ (typically the cumulative profit or total open equity
    of a financial trading strategy).

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


def roll_sharpe(series, period=252, win=0, cap=True):
    """ Compute rolling sharpe ratio.

    It is the rolling compouned annual returns divided by rolling annual
    volatility.

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
    ma[win:] = series[win:] / series[: -win]
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


if __name__ == '__main__':

    import doctest

    doctest.testmod()
