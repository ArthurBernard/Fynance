#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2018-12-14 19:11:40
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-16 11:45:46

""" Metric functons used in financial analysis. """

# Built-in packages

# External packages
import numpy as np

# Internal packages
from fynance.tools._wrappers import WrapperArray
from fynance.tools.metrics_cy import calmar_cy, drawdown_cy, mdd_cy, sharpe_cy
from fynance.tools.metrics_cy import log_sharpe_cy, roll_mdd_cy, roll_mad_cy_1d
from fynance.tools.metrics_cy import roll_mad_cy_2d
from fynance.tools.momentums_cy import smstd_cy
from fynance.tools.momentums import sma, ema, wma, smstd, emstd, wmstd

# TODO:
# - Append window size on rolling calmar
# - Append window size on rolling MDD
# - Append performance
# - Append rolling performance
# - verify and fix error to perf_strat, perf_returns, perf_index


__all__ = [
    'accuracy', 'annual_return', 'annual_volatility', 'calmar',
    'diversified_ratio', 'drawdown', 'mad', 'mdd', 'roll_calmar', 'roll_mad',
    'roll_mdd', 'roll_sharpe', 'roll_z_score', 'sharpe', 'perf_index',
    'perf_returns', 'z_score',
]

_handler_ma = {'s': sma, 'w': wma, 'e': ema}
_handler_mstd = {'s': smstd, 'w': wmstd, 'e': emstd}

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


def annual_return(series, period=252):
    """ Compute compouned annual return.

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    period : int, optional
        Number of period per year, default is 252 (trading days).

    Returns
    -------
    np.float64
        Value of compouned annual return.

    Examples
    --------
    Assume series of monthly prices:

    >>> series = np.array([100, 110, 80, 120, 160, 108])
    >>> print(round(annual_return(series, period=12), 4))
    0.1664

    See Also
    --------
    mdd, drawdown, sharpe, annual_volatility

    """
    T = series.size
    ret = series[-1] / series[0]

    return np.sign(ret) * np.float_power(
        np.abs(ret),
        period / T,
        dtype=np.float64
    ) - 1.


def annual_volatility(series, period=252):
    """ Compute compouned annual volatility.

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    period : int, optional
        Number of period per year, default is 252 (trading days).

    Returns
    -------
    np.float64
        Value of compouned annual volatility.

    Examples
    --------
    Assume series of monthly prices:

    >>> series = np.array([100, 110, 105, 110, 120, 108])
    >>> print(round(annual_volatility(series, period=12), 6))
    0.272172

    See Also
    --------
    mdd, drawdown, sharpe, annual_return

    """
    return np.sqrt(period) * np.std(series[1:] / series[:-1] - 1.)


def calmar(series, period=252):
    """ Compute the Calmar Ratio [1]_.

    Notes
    -----
    It is the compouned annual return over the Maximum DrawDown.

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

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Calmar_ratio

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

    Notes
    -----
    Diversification ratio, denoted D, is defined as the ratio of the
    portfolio's weighted average volatility to its overll volatility,
    developed by Choueifaty and Coignard [2]_.

    .. math:: D(P) = \frac{P' \Sigma}{\sqrt{P'VP}}

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
    .. [2] tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf

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
    variable [3]_ (typically the cumulative profit or total open equity of a
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
    .. [3] https://en.wikipedia.org/wiki/Drawdown_(economics)

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

    Compute the mean of the absolute value of the distance to the mean [4]_.

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
    .. [4] https://en.wikipedia.org/wiki/Average_absolute_deviation

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
    variable [5]_ (typically the cumulative profit or total open equity of a
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
    .. [5] https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

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


def perf_index(series, base=100.):
    """ Compute performance of prices or index values along time axis.

    Parameters
    ----------
    series : np.ndarray[ndim=1, dtype=np.float64]
        Time-series of prices or index values.
    base : float, optional
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
    >>> perf_index(series, base=100.)
    array([100., 120., 150., 140., 160., 180., 160.])

    """
    return base * series / series[0]


def perf_returns(returns, log=False, base=100.):
    """ Compute performance of returns along time axis.

    Parameters
    ----------
    returns : np.ndarray[ndim=1, dtype=np.float64]
        Time-series of returns.
    log : bool, optional
        Considers returns as log-returns if True. Default is False.
    base : float, optional
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
    >>> perf_returns(returns, base=100., log=False)
    array([100., 120., 150., 140., 160., 180., 160.])

    """
    series = np.cumsum(returns) + base

    if log:
        series = np.exp(series)

    return perf_index(series, base=base)


# TODO : finish perf strat metric (add reinvest option)
def perf_strat(underlying, signals=None, log=False, base=100.,
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
    base : float, optional
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
    >>> perf_strat(underlying, signals, base=100.)
    array([100., 120., 150., 150., 170., 190., 210.])

    # >>> perf_strat(underlying, signals, base=100., reinvest=True)
    # array([100., 120., ])

    """
    returns = np.zeros(underlying.shape)
    underlying *= base / underlying[0]
    returns[1:] = underlying[1:] - underlying[:-1]

    if signals is None:
        signals = np.ones(underlying.shape[0])

    series = returns * signals

    return perf_returns(series, log=log, base=base)


def sharpe(series, period=252, log=False):
    r""" Compute the Sharpe ratio [6]_.

    Notes
    -----
    It is computed as the total return over the volatility (we assume no
    risk-free rate) such that:

    .. math:: \text{Sharpe ratio} = \frac{E(r)}{\sqrt{Var(r)}}

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

    References
    ----------
    .. [6] https://en.wikipedia.org/wiki/Sharpe_ratio

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


def z_score(X, k=0, kind='s'):
    r""" Compute Z-score function.

    Notes
    -----
    Compute the z-score function for a specific average and standard deviation
    function such that:

    .. math:: z = \frac{X_t - \mu_t}{\sigma_t}

    Where :math:`\mu_t` is the average and :math:`\sigma_t` is the standard
    deviation.

    Parameters
    ----------
    X : np.ndarray[np.float64, ndim=1]
        Series of index, prices or returns.
    kind_ma : {'ema', 'sma', 'wma'}
        Kind of moving average/standard deviation, default is 'sma'.
        - Exponential moving average if 'ema'.
        - Simple moving average if 'sma'.
        - Weighted moving average if 'wma'.
    **kwargs
        Any parameters for the moving average function.

    Returns
    -------
    float
        Value of Z-score.

    Examples
    --------
    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> z_score(X, k=3, kind='e')
    -1.0443574118998766
    >>> z_score(X, k=3)
    -1.2247448713915896

    See Also
    --------
    roll_z_score, mdd, calmar, drawdown, sharpe

    """
    if k == 0:
        k = X.shape[0]

    avg = _handler_ma[kind.lower()](X, k=k)
    std = _handler_mstd[kind.lower()](X, k=k)

    std[std == 0.] = 1.
    z = (X - avg) / std

    return z[-1]


# =========================================================================== #
#                               Rolling metrics                               #
# =========================================================================== #

# TODO : rolling perf metric
# TODO : rolling diversified ratio

def roll_calmar(series, period=252.):
    """ Compute the rolling Calmar ratio [1]_.

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

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Calmar_ratio

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


@WrapperArray('dtype', 'axis')
def roll_mad(X, win=0, axis=0, dtype=None):
    """ Compute rolling Mean Absolut Deviation.

    Compute the moving average of the absolute value of the distance to the
    moving average [4]_.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time series (price, performance or index).
    win : int, optional
        Window size, default is 0.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Series of mean absolute deviation.

    References
    ----------
    .. [4] https://en.wikipedia.org/wiki/Average_absolute_deviation

    Examples
    --------
    >>> X = np.array([70, 100, 90, 110, 150, 80])
    >>> roll_mad(X, dtype=np.float64)
    array([ 0.        , 15.        , 11.11111111, 12.5       , 20.8       ,
           20.        ])
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> roll_mad(X, win=3, dtype=np.float64)
    array([ 0.        , 20.        , 13.33333333, 13.33333333, 26.66666667,
           26.66666667])

    See Also
    --------
    mad

    """
    if win < 2:
        win = X.shape[0]

    if len(X.shape) == 2:

        return np.asarray(roll_mad_cy_2d(X, int(win)))

    return np.asarray(roll_mad_cy_1d(X, int(win)))


def roll_mdd(series):
    """ Compute the rolling maximum drwdown.

    Where drawdown is the measure of the decline from a historical peak in
    some variable [5]_ (typically the cumulative profit or total open equity
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
    .. [5] https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

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
    """ Compute rolling sharpe ratio [6]_.

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

    References
    ----------
    .. [6] https://en.wikipedia.org/wiki/Sharpe_ratio

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


def roll_z_score(X, k=0, kind='s'):
    r""" Compute vector of rolling/moving Z-score function.

    Notes
    -----
    Compute for each observation the z-score function for a specific moving
    average function such that:

    .. math:: z = \frac{X - \mu_t}{\sigma_t}

    Where :math:`\mu_t` is the moving average and :math:`\sigma_t` is the
    moving standard deviation.

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Series of index, prices or returns.
    kind_ma : {'ema', 'sma', 'wma'}
        Kind of moving average/standard deviation, default is 'sma'.
        - Exponential moving average if 'ema'.
        - Simple moving average if 'sma'.
        - Weighted moving average if 'wma'.
    **kwargs
        Any parameters for the moving average function.

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Vector of Z-score at each period.

    Examples
    --------
    >>> series = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> roll_z_score(series, k=3, kind='e')
    array([ 0.        ,  1.41421356, -0.32444284,  1.30806216,  1.27096675,
           -1.04435741])
    >>> roll_z_score(series, k=3)
    array([ 0.        ,  1.        , -0.26726124,  1.22474487,  1.22474487,
           -1.22474487])

    See Also
    --------
    z_score, roll_mdd, roll_calmar, roll_mad, roll_sharpe

    """
    if k == 0:
        k = X.shape[0]

    avg = _handler_ma[kind.lower()](X, k=k)
    std = _handler_mstd[kind.lower()](X, k=k)

    std[std == 0.] = 1.
    z = (X - avg) / std

    return z


if __name__ == '__main__':

    import doctest

    doctest.testmod()
