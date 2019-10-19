#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2018-12-14 19:11:40
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-19 11:44:40

""" Metric functons used in financial analysis. """

# Built-in packages
from warnings import warn

# External packages
import numpy as np

# Internal packages
from fynance.tools._wrappers import WrapperArray
from fynance.tools.metrics_cy import calmar_cy, mdd_cy, sharpe_cy
from fynance.tools.metrics_cy import log_sharpe_cy, roll_mdd_cy, roll_mad_cy_1d
from fynance.tools.metrics_cy import roll_mad_cy_2d, drawdown_cy_1d
from fynance.tools.metrics_cy import drawdown_cy_2d
from fynance.tools.momentums_cy import smstd_cy
from fynance.tools.momentums import _sma, _ema, _wma, _smstd, _emstd, _wmstd

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

_handler_ma = {'s': _sma, 'w': _wma, 'e': _ema}
_handler_mstd = {'s': _smstd, 'w': _wmstd, 'e': _emstd}

# =========================================================================== #
#                                   Metrics                                   #
# =========================================================================== #


@WrapperArray('axis')
def accuracy(y_true, y_pred, sign=True, axis=0):
    r""" Compute the accuracy of prediction.

    Notes
    -----
    .. math::

        accuracy = \frac{right}{right + wrong}

    Parameters
    ----------
    y_true : np.ndarray[ndim=1 or 2, dtype]
        Vector of true series.
    y_pred : np.ndarray[ndim=1 or 2, dtype]
        Vector of predicted series.
    sign : bool, optional
        - If True then check sign accuracy (default).
        - Else check exact accuracy.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.

    Returns
    -------
    float or np.ndarray[ndim=1, float]
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
    R = np.sum(y_true == y_pred, axis=axis)

    # Check wrong answeres
    W = np.sum(y_true != y_pred, axis=axis)

    return R / (R + W)


@WrapperArray('dtype', 'axis')
def annual_return(X, period=252, axis=0, dtype=None):
    r""" Compute compouned annual returns of each `X`' series.

    The annualised return [1]_ is the process of converting returns on a whole
    period to returns per year.

    Notes
    -----
    Let T the number of time observations in `X`' series, the annual compouned
    returns is computed such that:

    .. math::

        annualReturn = sign(\frac{X_T}{X_0}) \time
        \frac{X_T}{X_0}^{\frac{period}{T - 1}} - 1

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of price, performance or index.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    dtype or np.ndarray[dtype, ndim=1]
        Values of compouned annual returns of each series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rate_of_return#Annualisation

    Examples
    --------
    Assume series of monthly prices:

    >>> X = np.array([100, 110, 80, 120, 160, 125, 108]).astype(np.float64)
    >>> print(round(annual_return(X, period=12), 4))
    0.1664
    >>> X = np.array([[100, 110], [80, 120], [160, 108]]).astype(np.float64)
    >>> annual_return(X, period=12)
    array([15.777216  , -0.10425081])

    See Also
    --------
    mdd, drawdown, sharpe, annual_volatility

    """
    return _annual_return(X, period=period)


def _annual_return(X, period):
    if (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    ret = X[-1] / X[0]

    if (ret < 0).any():

        raise ValueError('initial value X[0] and final value X[T] must \
            be of the same sign.')

    T = X.shape[0] - 1
    ret = np.abs(ret)
    sign = np.sign(X[0])

    return sign * np.float_power(ret, period / T, dtype=np.float64) - 1.


@WrapperArray('dtype', 'axis', 'null')
def annual_volatility(X, period=252, log=True, axis=0, dtype=None):
    r""" Compute the annualized volatility of each `X`' series.

    In finance, volatility is the degree of variation of a trading price
    series over time as measured by the standard deviation of logarithmic
    returns [2]_.

    Notes
    -----
    Let :math:`Var` the variance function of a random variable:

    .. math::

        annualVolatility = \sqrt{period \times Var_t(R)} \\
        \text{where, }R =
        \begin{cases}ln(\frac{X_{1:T}}{X_{0:T-1}}) \text{ if log=True}\\
                    \frac{X_{1:T}}{X_{0:T-1}} - 1 \text{ otherwise} \\
        \end{cases}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of price, performance or index.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    log : bool, optional
        - If True then logarithmic returns are computed.
        - Else then returns in percentage are computed.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    dtype or np.ndarray([dtype, ndim=1])
        Values of annualized volatility for each series.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Volatility_(finance)

    Examples
    --------
    Assume series of monthly prices:

    >>> X = np.array([100, 110, 105, 110, 120, 108]).astype(np.float64)
    >>> print(round(annual_volatility(X, period=12, log=True), 6))
    0.272321
    >>> annual_volatility(X.reshape([6, 1]), period=12, log=False)
    array([0.27217177])

    See Also
    --------
    mdd, drawdown, sharpe, annual_return

    """
    if log:
        R = np.log(X[1:] / X[:-1])

    else:
        R = X[1:] / X[:-1] - 1.

    return np.sqrt(period) * np.std(R, axis=axis)


@WrapperArray('dtype', 'axis')
def calmar(X, raw=False, period=252, axis=0, dtype=None):
    r""" Compute the Calmar Ratio [3]_ for each `X`' series.

    Notes
    -----
    It is the compouned annual return
    (:func:`~fynance.tools.metrics.annual_return`) over the maximum drawdown
    (:func:`~fynance.tools.metrics.mdd`). Let :math:`T` the number of time
    observations, DD the vector of drawdown:

    .. math::

        calmarRatio = \frac{annualReturn}{MaxDD} \\
        annualReturn = sign(\frac{X_T}{X_0}) \time
        \frac{X_T}{X_0}^{\frac{period}{T - 1}} - 1 \\
        maxDD = max(DD) \\
        \text{where, } DD_t =
        \begin{cases}max(X_{0:t}) - X_t \text{ if raw=True} \\
                     1 - \frac{X_t}{max(X_{0:t})} \text{ otherwise} \\
        \end{cases}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series price, performance or index.
    raw : bool, optional
        - If True then compute the raw drawdown.
        - Else (default) compute the drawdown in percentage.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    dtype or np.ndarray([dtype, ndim=1])
        Values of Calmar ratio for each series.

    References
    ----------
    .. [3] https://en.wikipedia.org/wiki/Calmar_ratio

    Examples
    --------
    Assume a series of monthly prices:

    >>> X = np.array([70, 100, 80, 120, 160, 105, 80]).astype(np.float64)
    >>> calmar(X, period=12)
    0.6122448979591835
    >>> calmar(X.reshape([7, 1]), period=12)
    array([0.6122449])

    See Also
    --------
    mdd, drawdown, sharpe, roll_calmar

    """
    # TODO: check if cython function is necessary
    return _annual_return(X, period) / _drawdown(X, raw).max(axis=axis)


@WrapperArray('axis')
def diversified_ratio(X, W=None, std_method='std', axis=0):
    r""" Compute diversification ratio of a portfolio.

    Notes
    -----
    Diversification ratio, denoted D, is defined as the ratio of the
    portfolio's weighted average volatility to its overll volatility,
    developed by Choueifaty and Coignard [4]_.

    .. math:: D(P) = \frac{P' \Sigma}{\sqrt{P'VP}}

    With :math:`\Sigma` vector of asset volatilities, :math:`P` vector of
    weights of asset of portfolio, and :math:`V` matrix of variance-covariance
    of these assets.

    Parameters
    ----------
    X : np.ndarray[ndim=2, dtype=np.float64] of shape (T, N)
        Portfolio matrix of N assets and T time periods, each column
        correspond to one series of prices.
    W : np.array[ndim=1 or 2, dtype=np.float64] of size N, optional
        Vector of weights, default is None it means it will equaly weighted.
    std_method : str, optional /!\ Not yet implemented /!\
        Method to compute variance vector and covariance matrix.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.

    Returns
    -------
    np.float64
        Value of diversification ratio of the portfolio.

    References
    ----------
    .. [4] tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf

    """
    # TODO : check efficiency
    #        append examples
    T, N = X.shape

    if W is None:
        W = np.ones([N, 1]) / N
    else:
        W = W.reshape([N, 1])

    sigma = np.std(X, axis=0).reshape([N, 1])
    V = np.cov(X, rowvar=False, bias=True).reshape([N, N])

    return (W.T @ sigma) / np.sqrt(W.T @ V @ W)


@WrapperArray('dtype', 'axis')
def drawdown(X, raw=False, axis=0, dtype=None):
    r""" Measures the drawdown of each `X`' series.

    Function to compute measure of the decline from a historical peak in some
    variable [5]_ (typically the cumulative profit or total open equity of a
    financial trading strategy).

    Notes
    -----
    Let DD the drawdown vector:

    .. math::

        DD_t = \begin{cases}max(X_{0:t}) - X_t \text{ if raw=True} \\
                            1 - \frac{X_t}{max(X_{0:t})} \text{ otherwise} \\
        \end{cases}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices, performances or index. Must be positive values.
    raw : bool, optional
        - If True then compute the raw drawdown.
        - Else (default) compute the drawdown in percentage.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Series of drawdown for each series.

    References
    ----------
    .. [5] https://en.wikipedia.org/wiki/Drawdown_(economics)

    Examples
    --------
    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> drawdown(X)
    array([0. , 0. , 0.2, 0. , 0. , 0.5])
    >>> drawdown(X.reshape([6, 1])).T
    array([[0. , 0. , 0.2, 0. , 0. , 0.5]])
    >>> drawdown(X, raw=True)
    array([ 0.,  0., 20.,  0.,  0., 80.])

    See Also
    --------
    mdd, calmar, sharpe, roll_mdd

    """
    return _drawdown(X, raw)


def _drawdown(X, raw):
    if (X[0] == 0).any() and not raw:

        warn('Cannot compute drawdown in percentage without initial values \
            X[0] strictly positive.')
        raw = True

    if len(X.shape) == 2:

        return np.asarray(drawdown_cy_2d(X, int(raw)))

    return np.asarray(drawdown_cy_1d(X, int(raw)))


@WrapperArray('dtype')
def mad(X, axis=0, dtype=None):
    """ Compute the Mean Absolute Deviation of each `X`' series.

    Compute the mean of the absolute value of the distance to the mean [6]_.

    Parameters
    ----------
    X : np.ndarray[np.dtype, ndim=1 or 2]
        Time-series of prices, performances or index.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    dtype or np.ndarray[dtype, ndim=1]
        Values of mean absolute deviation of each series.

    References
    ----------
    .. [6] https://en.wikipedia.org/wiki/Average_absolute_deviation

    Examples
    --------
    >>> X = np.array([70., 100., 90., 110., 150., 80.])
    >>> mad(X)
    20.0

    See Also
    --------
    roll_mad

    """
    # TODO : make cython function or not ?
    return np.mean(np.abs(X - np.mean(X, axis=axis)), axis=axis)


@WrapperArray('dtype', 'axis')
def mdd(X, raw=False, axis=0, dtype=None):
    r""" Compute the maximum drawdown for each `X`' series.

    Drawdown (:func:~`fynance.tools.metrics.drawdown`) is the measure of the
    decline from a historical peak in some variable [5]_ (typically the
    cumulative profit or total open equity of a financial trading strategy).

    Notes
    -----
    Let DD the drawdown vector:

    .. math::

        maxDD = max(DD) \\
        \text{where, } DD_t =
        \begin{cases}max(X_{0:t}) - X_t \text{ if raw=True} \\
                     1 - \frac{X_t}{max(X_{0:t})} \text{ otherwise} \\
        \end{cases}

    Parameters
    ----------
    X : np.ndarray[np.dtype, ndim=1 or 2]
        Time-series of prices, performances or index.
    raw : bool, optional
        - If True then compute the raw drawdown.
        - Else (default) compute the drawdown in percentage.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    dtype or np.ndarray[dtype, ndim=1]
        Value of Maximum DrawDown for each series.

    References
    ----------
    .. [5] https://en.wikipedia.org/wiki/Drawdown_(economics)

    Examples
    --------
    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> mdd(X)
    0.5
    >>> mdd(X.reshape([6, 1]))
    array([0.5])

    See Also
    --------
    drawdown, calmar, sharpe, roll_mdd

    """
    return _drawdown(X, raw).max(axis=axis)


def perf_index(X, base=100.):
    """ Compute performance of prices or index values along time axis.

    Parameters
    ----------
    X : np.ndarray[ndim=1, dtype=np.float64]
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
    >>> X = np.array([10., 12., 15., 14., 16., 18., 16.])
    >>> perf_index(X, base=100.)
    array([100., 120., 150., 140., 160., 180., 160.])

    """
    return base * X / X[0]


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
    X = np.cumsum(returns) + base

    if log:
        X = np.exp(X)

    return perf_index(X, base=base)


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

    X = returns * signals

    return perf_returns(X, log=log, base=base)


@WrapperArray('dtype', 'axis')
def sharpe(X, period=252, log=False, axis=0, dtype=None):
    r""" Compute the Sharpe ratio [7]_ for each `X`' series.

    Notes
    -----
    It is computed as the total return over the volatility (we assume no
    risk-free rate) such that:

    .. math:: \text{Sharpe ratio} = \frac{E(r)}{\sqrt{Var(r)}}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices, performances or index.
    period : int, optional
        Number of period per year, default is 252 (trading days).
    log : bool, optional
        If true compute sharpe with the formula for log-returns, default
        is False.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    dtype or np.ndarray[dtype, ndim=1]
        Value of Sharpe ratio for each series.

    References
    ----------
    .. [7] https://en.wikipedia.org/wiki/Sharpe_ratio

    Examples
    --------
    Assume a series X of monthly prices:

    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> sharpe(X, period=12)
    0.22494843872918127
    >>> sharpe(X.reshape([6, 1]), period=12)
    array([0.22494844])

    See Also
    --------
    mdd, calmar, drawdown, roll_sharpe

    """
    # TODO : check efficiency of cython function
    #        append risk free rate
    sharpe_func = log_sharpe_cy if log else sharpe_cy

    if len(X.shape) == 2:
        output = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            output[i] = sharpe_func(X[:, i], float(period))

        return output

    return sharpe_func(X, float(period))


@WrapperArray('dtype', 'axis', 'window')
def z_score(X, w=0, kind='s', axis=0, dtype=None):
    r""" Compute the Z-score of each `X`' series.

    Notes
    -----
    Compute the z-score function for a specific average and standard deviation
    function such that:

    .. math:: z = \frac{X_t - \mu_t}{\sigma_t}

    Where :math:`\mu_t` is the average and :math:`\sigma_t` is the standard
    deviation.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Series of index, prices or returns.
    w : int, optional
        Size of the lagged window of the moving averages, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    kind : {'e', 's', 'w'}
        - If 'e' then use exponential moving average, see
          :func:`~fynance.tools.momentums.ema` for details.
        - If 's' (default) then use simple moving average, see
          :func:`~fynance.tools.momentums.sma` for details.
        - If 'w' then use weighted moving average, see
          :func:`~fynance.tools.momentums.wma` for details.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    dtype or np.ndarray[dtype, ndim=1]
        Value of Z-score for each series.

    Examples
    --------
    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> z_score(X, w=3, kind='e')
    -1.0443574118998766
    >>> z_score(X, w=3)
    -1.2247448713915896
    >>> z_score(X.reshape([6, 1]), w=3)
    array([-1.22474487])

    See Also
    --------
    roll_z_score, mdd, calmar, drawdown, sharpe

    """
    if kind == 'e':
        w = 1 - 2 / (1 + w)

    avg = _handler_ma[kind.lower()](X, w)
    std = _handler_mstd[kind.lower()](X, w)

    std[std == 0.] = 1.
    z = (X - avg) / std

    return z[-1]


# =========================================================================== #
#                               Rolling metrics                               #
# =========================================================================== #

# TODO : rolling perf metric
# TODO : rolling diversified ratio

def roll_calmar(X, period=252.):
    """ Compute the rolling Calmar ratio [1]_.

    It is the compouned annual return over the rolling Maximum DrawDown.

    Parameters
    ----------
    X : np.ndarray[np.float64, ndim=1]
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

    >>> X = np.array([70, 100, 80, 120, 160, 80])
    >>> roll_calmar(X, period=12)
    array([ 0.        ,  0.        ,  3.52977926, 20.18950437, 31.35989887,
            0.6122449 ])

    See Also
    --------
    roll_mdd, roll_sharpe, calmar

    """
    # Set variables
    X = np.asarray(X, dtype=np.float64).flatten()
    T = X.size
    t = np.arange(1., T + 1., dtype=np.float64)

    # Compute roll Returns
    ret = X / X[0]
    annual_return = np.sign(ret) * np.float_power(
        np.abs(ret), period / t, dtype=np.float64) - 1.

    # Compute roll MaxDrawDown
    roll_maxdd = roll_mdd_cy(X)

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


def roll_mdd(X):
    """ Compute the rolling maximum drwdown.

    Where drawdown is the measure of the decline from a historical peak in
    some variable [5]_ (typically the cumulative profit or total open equity
    of a financial trading strategy).

    Parameters
    ----------
    X : np.ndarray[np.float64, ndim=1]
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
    >>> X = np.array([70, 100, 80, 120, 160, 80])
    >>> roll_mdd(X)
    array([0. , 0. , 0.2, 0.2, 0.2, 0.5])

    See Also
    --------
    mdd, roll_calmar, roll_sharpe, drawdown

    """
    X = np.asarray(X, dtype=np.float64).flatten()

    return roll_mdd_cy(X)


def roll_sharpe(X, period=252, win=0, cap=True):
    """ Compute rolling sharpe ratio [6]_.

    It is the rolling compouned annual returns divided by rolling annual
    volatility.

    Parameters
    ----------
    X : np.ndarray[dtype=np.float64, ndim=1]
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

    >>> X = np.array([70, 100, 80, 120, 160, 80])
    >>> roll_sharpe(X, period=12)
    array([0.        , 0.        , 0.77721579, 3.99243019, 6.754557  ,
           0.24475518])

    See Also
    --------
    roll_calmar, sharpe, roll_mdd

    """
    # Setting inputs
    X = np.asarray(X, dtype=np.float64).flatten()
    T = X.size
    t = np.arange(1., T + 1., dtype=np.float64)

    if win < 2:
        win = T

    t[t > win] = win + 1.
    ret = np.zeros([T], dtype=np.float64)
    ret[1:] = X[1:] / X[:-1] - 1.

    # Compute rolling perf
    ma = X / X[0]
    ma[win:] = X[win:] / X[: -win]
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


def roll_z_score(X, w=0, kind='s'):
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
    np.ndarray[np.float64, ndim=1]
        Vector of Z-score at each period.

    Examples
    --------
    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> roll_z_score(X, w=3, kind='e')
    array([ 0.        ,  1.41421356, -0.32444284,  1.30806216,  1.27096675,
           -1.04435741])
    >>> roll_z_score(X, w=3)
    array([ 0.        ,  1.        , -0.26726124,  1.22474487,  1.22474487,
           -1.22474487])

    See Also
    --------
    z_score, roll_mdd, roll_calmar, roll_mad, roll_sharpe

    """
    if w == 0:
        w = X.shape[0]

    if kind == 'e':
        w = 1 - 2 / (1 + w)

    avg = _handler_ma[kind.lower()](X, w)
    std = _handler_mstd[kind.lower()](X, w)

    std[std == 0.] = 1.
    z = (X - avg) / std

    return z


if __name__ == '__main__':

    import doctest

    doctest.testmod()
