#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2018-12-14 19:11:40
# @Last modified by: ArthurBernard
# @Last modified time: 2020-01-24 15:57:17

""" Performance and risk metrics for financial analysis.

Compute risk-adjusted returns, drawdown statistics and other summary
indicators commonly used to evaluate strategies and portfolios. All
functions accept a 1-D or 2-D array of prices/returns and return the
metric along the time axis.

Main entry points
-----------------
- :func:`annual_return`, :func:`annual_volatility` — annualized
  return and volatility from a price series.
- :func:`sharpe`, :func:`calmar`, :func:`diversified_ratio` —
  risk-adjusted performance ratios.
- :func:`mdd`, :func:`drawdown` — maximum drawdown and drawdown path.
- :func:`z_score`, :func:`accuracy` — statistical helpers.

"""

from __future__ import annotations

# Built-in packages
from warnings import warn

# External packages
import numpy as np
from numpy.typing import NDArray

from fynance._exceptions import ArraySizeError

# Internal packages
from fynance._wrappers import WrapperArray
from fynance.features.metrics_cy import *
from fynance.features.momentums import _ema, _emstd, _sma, _smstd, _wma, _wmstd

# TODO:
# - Append performance
# - Append rolling performance
# - verify and fix error to perf_strat, perf_returns, perf_index


__all__ = [
    'accuracy', 'annual_return', 'annual_volatility', 'calmar',
    'directional_accuracy', 'diversified_ratio', 'drawdown', 'mad', 'mdd',
    'roll_annual_return', 'roll_annual_volatility', 'roll_calmar',
    'roll_drawdown', 'roll_mad', 'roll_mdd', 'roll_sharpe', 'roll_z_score',
    'sharpe', 'sortino', 'perf_index', 'perf_returns', 'z_score',
]

_handler_ma = {'s': _sma, 'w': _wma, 'e': _ema}
_handler_mstd = {'s': _smstd, 'w': _wmstd, 'e': _emstd}

# =========================================================================== #
#                                   Metrics                                   #
# =========================================================================== #


@WrapperArray('axis')
def accuracy(y_true: NDArray, y_pred: NDArray, sign: bool = True, axis: int = 0) -> float:
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


@WrapperArray('dtype', 'axis', 'ddof', min_size=2)
def annual_return(X: NDArray, period: int = 252, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute compouned annual returns of each `X`' series.

    The annualised return [1]_ is the process of converting returns on a whole
    period to returns per year.

    Notes
    -----
    Let T the number of timeframes in `X`' series, the annual compouned returns
    is computed such that:

    .. math::

        annualReturn = \frac{X_T}{X_1}^{\frac{period}{T}} - 1

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
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``T - ddof``, where ``T`` represents the number of elements in time
        axis. Default is 0.

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

    >>> X = np.array([100, 110, 80, 120, 160, 108]).astype(np.float64)
    >>> print(round(annual_return(X, period=12), 4))
    0.1664
    >>> X = np.array([[100, 110], [80, 120], [160, 108]]).astype(np.float64)
    >>> annual_return(X, period=12, ddof=1)
    array([15.777216  , -0.10425081])

    See Also
    --------
    mdd, drawdown, sharpe, annual_volatility

    """
    if ddof >= X.shape[0]:

        raise ValueError("degree of freedom {} is greater than size {} of X "
                         "in axis {}".format(ddof, X.shape[axis], axis))

    return _annual_return(X, period, ddof)


def _annual_return(X, period, ddof):
    if (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    ret = X[-1] / X[0]
    T = X.shape[0]

    if (ret < 0).any():

        raise ValueError('initial value X[0] and final value X[-1] must '
                         'be of the same sign.')

    sign = np.sign(X[0])
    power = period / (T - ddof)

    return sign * np.float_power(ret, power, dtype=np.float64) - 1.


@WrapperArray('dtype', 'axis', 'null', 'ddof', min_size=2)
def annual_volatility(X: NDArray, period: int = 252, log: bool = True, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute the annualized volatility of each `X`' series.

    In finance, volatility is the degree of variation of a trading price
    series over time as measured by the standard deviation of logarithmic
    returns [2]_.

    Notes
    -----
    Let :math:`Var` the variance function of a random variable:

    .. math::

        annualVolatility = \sqrt{period \times Var(R_{1:T})}

    Where, :math:`R_1 = 0` and :math:`R_{2:T} = \begin{cases}ln(\frac{X_{2:T}}
    {X_{1:T-1}}) \text{, if log=True} \\ \frac{X_{2:T}}{X_{1:T-1}} - 1 \text{,
    otherwise} \\ \end{cases}`

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
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``T - ddof``, where ``T`` represents the number of elements in time
        axis. Default is 0.

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
    >>> annual_volatility(X, period=12, log=True, ddof=1)
    0.2731896268610321
    >>> annual_volatility(X.reshape([6, 1]), period=12, log=False)
    array([0.24961719])

    See Also
    --------
    mdd, drawdown, sharpe, annual_return

    """
    return _annual_volatility(X, period, log, axis, ddof)


def _annual_volatility(X, period, log, axis, ddof):
    R = np.zeros(X.shape)

    if log:
        R[1:] = np.log(X[1:] / X[:-1])

    else:
        R[1:] = X[1:] / X[:-1] - 1.

    return np.sqrt(period) * np.std(R, axis=axis, ddof=ddof)


def _annual_downside_volatility(X, period, log, axis, ddof):
    R = np.zeros(X.shape)

    if log:
        R[1:] = np.log(X[1:] / X[:-1])

    else:
        R[1:] = X[1:] / X[:-1] - 1.

    R_neg = np.where(R < 0, R, 0.)

    return np.sqrt(period) * np.std(R_neg, axis=axis, ddof=ddof)


@WrapperArray('dtype', 'axis', 'ddof', min_size=2)
def calmar(X: NDArray, period: int = 252, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute the Calmar Ratio for each `X`' series.

    Notes
    -----
    Calmar ratio [3]_ is the compouned annual return
    (:func:`~fynance.features.metrics.annual_return`) over the maximum drawdown
    (:func:`~fynance.features.metrics.mdd`). Let :math:`T` the number of time
    observations, DD the vector of drawdown:

    .. math::

        calmarRatio = \frac{annualReturn}{MDD}

    With, :math:`annualReturn = \frac{X_T}{X_1}^{\frac{period}{T}} - 1` and
    :math:`MDD = max(DD_{1:T})`.

    Where, :math:`DD_t = 1 - \frac{X_t}{max(X_{1:t})}`,
    :math:`\forall t \in [1:T]`.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series price, performance or index.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``T - ddof``, where ``T`` represents the number of elements in time
        axis. Default is 0.

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
    >>> calmar(X, period=12, ddof=1)
    array(0.6122449)
    >>> calmar(X.reshape([7, 1]), period=12)
    array([0.51446018])

    See Also
    --------
    mdd, drawdown, sharpe, roll_calmar

    """
    ret = _annual_return(X, period, ddof)
    dd = _drawdown(X, False)
    mdd = np.max(dd, axis=axis)
    calmar = np.zeros(ret.shape)
    slice_bool = (mdd != 0)
    calmar[slice_bool] = ret[slice_bool] / mdd[slice_bool]

    return calmar


@WrapperArray('axis')
def diversified_ratio(X: NDArray, W: NDArray | None = None, std_method: str = 'std', axis: int = 0) -> float:
    """ Compute diversification ratio of a portfolio.

    Notes
    -----
    Diversification ratio, denoted D, is defined as the ratio of the
    portfolio's weighted average volatility to its overll volatility,
    developed by Choueifaty and Coignard [4]_.

    .. math:: D(P) = \\frac{P' \\Sigma}{\\sqrt{P'VP}}

    With :math:`\\Sigma` vector of asset volatilities, :math:`P` vector of
    weights of asset of portfolio, and :math:`V` matrix of variance-covariance
    of these assets.

    Parameters
    ----------
    X : np.ndarray[ndim=2, dtype=np.float64] of shape (T, N)
        Portfolio matrix of N assets and T time periods, each column
        correspond to one series of prices.
    W : np.array[ndim=1 or 2, dtype=np.float64] of size N, optional
        Vector of weights, default is None it means it will equaly weighted.
    std_method : str, optional /!\\ Not yet implemented /!\
        Method to compute variance vector and covariance matrix.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.

    Returns
    -------
    np.float64
        Value of diversification ratio of the portfolio.

    References
    ----------
    .. [4] `Choueifaty, Y., and Coignard, Y., 2008, Toward Maximum \
    Diversification. <https://www.tobam.fr/wp-content/uploads/2014/12/\
    TOBAM-JoPM-Maximum-Div-2008.pdf>`_

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
def drawdown(X: NDArray, raw: bool = False, axis: int = 0, dtype=None) -> NDArray:
    r""" Measures the drawdown of each `X`' series.

    Function to compute measure of the decline from a historical peak in some
    variable [5]_ (typically the cumulative profit or total open equity of a
    financial trading strategy).

    Notes
    -----
    Let DD the drawdown vector, :math:`\forall t \in [1:T]`:

    .. math::

        DD_t = \begin{cases}max(X_{1:t}) - X_t \text{, if raw=True} \\
                            1 - \frac{X_t}{max(X_{1:t})} \text{, otherwise} \\
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

        warn(
            'Cannot compute drawdown in percentage without initial values '
            'X[0] strictly positive.',
            category=UserWarning,
            stacklevel=2,
        )
        raw = True

    if len(X.shape) == 2:

        return np.asarray(drawdown_cy_2d(X, int(raw)))

    return np.asarray(drawdown_cy_1d(X, int(raw)))


@WrapperArray('dtype')
def mad(X: NDArray, axis: int = 0, dtype=None) -> NDArray:
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
    return np.mean(np.abs(X.T - np.mean(X, axis=axis)).T, axis=axis)


@WrapperArray('dtype', 'axis')
def mdd(X: NDArray, raw: bool = False, axis: int = 0, dtype=None) -> NDArray:
    r""" Compute the maximum drawdown for each `X`' series.

    Maximum peak-to-trough decline observed over the full series. A
    standard tail-risk indicator: it captures the worst loss an
    investor would have endured, regardless of horizon. Reported in
    relative terms by default (fraction of peak); use ``raw=True`` for
    an absolute decline. For the full drawdown path use
    :func:`drawdown`; combined with annual return, it gives the Calmar
    ratio (:func:`calmar`).

    Drawdown (:func:~`fynance.features.metrics.drawdown`) is the measure of the
    decline from a historical peak in some variable [5]_ (typically the
    cumulative profit or total open equity of a financial trading strategy).

    Notes
    -----
    Let DD the drawdown vector:

    .. math::

        MDD = max(DD_{1:T})

    Where, :math:`DD_t = \begin{cases}max(X_{1:t})
    - X_t \text{, if raw=True} \\ 1 - \frac{X_t}{max(X_{1:t})} \text{,
    otherwise} \\ \end{cases}`, :math:`\forall t \in [1:T]`.

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


@WrapperArray('dtype', 'axis')
def perf_index(X: NDArray, base: float = 100., axis: int = 0, dtype=None) -> NDArray:
    """ Compute performance of prices or index values along time axis.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices or index values.
    base : float, optional
        Initial value for measure the performance, default is 100.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
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


@WrapperArray('dtype', 'axis')
def perf_returns(R: NDArray, kind: str = 'raw', base: float = 100., axis: int = 0, dtype=None) -> NDArray:
    """ Compute performance of returns along time axis.

    Parameters
    ----------
    R : np.ndarray[dtype, ndim=1 or 2]
        Time-series of returns.
    kind : {'raw', 'log', 'pct'}
        - If `'raw'` (default), then considers returns as following :math:`R_t
          = X_t - X_{t-1}`.
        - If `'log'`, then considers returns as following :math:`R_t =
          log(\\frac{X_t}{X_{t-1}})`.
        - If `'pct'`, then considers returns as following :math:`R_t =
          \\frac{X_t - X_{t-1}}{X_{t-1}}`.
    base : float, optional
        Initial value for measure the performance, default is 100.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Performances along time axis.

    See Also
    --------
    perf_index, perf_strat

    Examples
    --------
    >>> R = np.array([0., 20., 30., -10., 20., 20., -20.])
    >>> perf_returns(R, base=100.)
    array([100., 120., 150., 140., 160., 180., 160.])

    """
    if kind.lower() == 'raw':
        X = base + np.cumsum(R, axis=axis)

    elif kind.lower() == 'log':
        X = base * np.cumprod(np.exp(R), axis=axis)

    elif kind.lower() == 'pct':
        X = base * np.cumprod(R + 1., axis=axis)

    else:

        raise ValueError("unkwnown kind {} of returns, only {'raw', 'log',"  # noqa: F524
                         "'pct'} are supported".format(kind))

    return perf_index(X, base=base, axis=axis, dtype=dtype)


@WrapperArray('dtype', 'axis')
def perf_strat(X: NDArray, S: NDArray | None = None, base: float = 100., axis: int = 0, dtype=None, reinvest: bool = False) -> NDArray:
    """ Compute the performance of strategies for each `X`' series.

    With respect to this underlying and signal series along time axis.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices or index values.
    S : np.ndarray[dtype, ndim=1 or 2]
        Time-series of signals, if `None` considering a long only position.
        ``S`` array must have the same shape than ``X``. Default is None.
    base : float, optional
        Initial value for measure the performance, default is 100.
    reinvest : bool, optional
        - If True, then reinvest profit to compute the performance.
        - Otherwise (default), compute the performance without reinvesting.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Performances along time axis.

    See Also
    --------
    perf_returns, perf_index

    Examples
    --------
    >>> X = np.array([10., 12., 15., 14., 16., 18., 16.])
    >>> S = np.array([1., 1., 1., 0., 1., 1., -1.])
    >>> perf_strat(X, S, base=100.)
    array([100., 120., 150., 150., 170., 190., 210.])
    >>> perf_strat(X, S, base=100., reinvest=True)
    array([100.        , 120.        , 150.        , 150.        ,
           171.42857143, 192.85714286, 214.28571429])

    """
    if S is None:
        S = np.ones(X.shape)

    elif S.ndim < X.ndim and S.shape[0] == X.shape[0]:
        S = S.reshape(S.shape + (1,) * (X.ndim - S.ndim))

    elif S.shape == X.shape:
        pass

    else:
        raise ValueError('S and X could not be broadcast, must have the same '
                         'shape or axis {} of same size and S.ndim < X.ndim '
                         '(but S:{} and X:{})'.format(axis, S.shape, X.shape))

    R = np.zeros(X.shape)
    X = base * X / X[0]

    if not reinvest:
        kind = 'raw'
        R[1:] = X[1:] - X[:-1]

    elif reinvest:
        kind = 'pct'
        R[1:] = X[1:] / X[:-1] - 1

    return perf_returns(R * S, base=base, kind=kind, axis=axis, dtype=dtype)


@WrapperArray('dtype', 'axis')
def returns_strat(X: NDArray, S: NDArray | None = None, kind: str = 'pct', base: float = 100., axis: int = 0, dtype=None) -> NDArray:
    r""" Compute the returns of strategies for each `X`' series.

    With respect to this underlying and signal series along time axis.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices or index values.
    S : np.ndarray[dtype, ndim=1 or 2]
        Time-series of signals, if `None` considering a long only position.
        ``S`` array must have the same shape than ``X``. Default is None.
    kind : {'raw', 'pct'}
        - If `'raw'`, then considers returns as following :math:`R_t
          = X_t - X_{t-1}`.
        - If `'pct'` (default), then considers returns as following
          :math:`R_t = \frac{X_t - X_{t-1}}{X_{t-1}}`.
    base : float, optional
        Initial value for measure the returns, default is 100. Relevant only if
        ``kind='raw'``.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Returns along time axis.

    See Also
    --------
    perf_returns, perf_index

    Examples
    --------
    >>> X = np.array([10., 12., 15., 14., 16., 18., 16.])
    >>> S = np.array([1., 1., 1., 0., 1., 1., -1.])
    >>> returns_strat(X, S, base=10., kind='raw')
    array([ 0.,  2.,  3., -0.,  2.,  2.,  2.])
    >>> returns_strat(X, S)
    array([ 0.        ,  0.2       ,  0.25      , -0.        ,  0.14285714,
            0.125     ,  0.11111111])

    """
    if S is None:
        S = np.ones(X.shape)

    elif S.ndim < X.ndim and S.shape[0] == X.shape[0]:
        S = S.reshape(S.shape + (1,) * (X.ndim - S.ndim))

    elif S.shape == X.shape:
        pass

    else:
        raise ValueError('S and X could not be broadcast, must have the same '
                         'shape or axis {} of same size and S.ndim < X.ndim '
                         '(but S:{} and X:{})'.format(axis, S.shape, X.shape))

    R = np.zeros(X.shape)
    X = base * X / X[0]

    if kind == 'raw':
        R[1:] = X[1:] - X[:-1]

    elif kind == 'pct':
        R[1:] = X[1:] / X[:-1] - 1

    return R * S


@WrapperArray('dtype', 'axis', 'null', 'ddof', min_size=2)
def sharpe(X: NDArray, rf: float = 0, period: int = 252, log: bool = False, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute the Sharpe ratio for each `X`' series.

    Annualized excess return per unit of volatility — the most widely
    used risk-adjusted performance metric. Higher is better; ratios
    above 1 are usually considered good and above 2 excellent for
    long-horizon strategies. Note that the Sharpe ratio penalizes both
    upside and downside volatility symmetrically; use the Sortino or
    Calmar ratio (:func:`calmar`) when only downside risk should be
    penalized.

    The ``period`` argument controls annualization (252 for daily
    trading data, 12 for monthly, etc.). For a rolling estimate, see
    :func:`roll_sharpe`.

    Notes
    -----
    Sharpe ratio [7]_ is computed as the annualized expected returns
    (:func:`~annual_return`) minus the risk-free rate (noted :math:`rf`) over
    the annualized volatility of returns (:func:`~annual_volatility`) such
    that:

    .. math::

        sharpeRatio = \frac{E(R) - rf}{\sqrt{period \times Var(R)}} \\ \\

    where, :math:`R_1 = 0` and :math:`R_{2:T} = \begin{cases}ln(\frac{X_{2:T}}
    {X_{1:T-1}}) \text{, if log=True}\\ \frac{X_{2:T}}{X_{1:T-1}} - 1 \text{,
    otherwise} \\ \end{cases}`

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices, performances or index.
    rf : float, optional
        Means the annualized risk-free rate, default is 0.
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
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``T - ddof``, where ``T`` represents the number of elements in time
        axis. Default is 0.

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
    0.24475518072327812
    >>> sharpe(X.reshape([6, 1]), period=12)
    array([0.24475518])

    See Also
    --------
    mdd, calmar, drawdown, roll_sharpe

    """
    ret = _annual_return(X, period, ddof)
    vol = _annual_volatility(X, period, log, axis, ddof)
    if (vol == 0.).any():
        res = ret - rf
        res[vol == 0.] = np.inf
        res[vol != 0.] = res[vol != 0.] / vol[vol != 0.]

        return res

    return (ret - rf) / vol


@WrapperArray('dtype', 'axis', 'null', 'ddof', min_size=2)
def sortino(
    X: NDArray, rf: float = 0, period: int = 252, log: bool = False,
    axis: int = 0, dtype=None, ddof: int = 0,
) -> NDArray:
    r""" Compute the Sortino ratio for each `X`' series.

    Annualized excess return per unit of *downside* volatility. Unlike the
    Sharpe ratio (:func:`sharpe`), only negative returns contribute to the
    denominator, so strategies that generate frequent large gains are not
    penalized for their upside variance.

    Notes
    -----
    The Sortino ratio is computed as the annualized expected return minus
    the risk-free rate divided by the annualized downside deviation:

    .. math::

        sortinoRatio = \frac{E(R) - rf}{\sqrt{period \times Var(R^{-})}}

    where :math:`R^{-}_t = \min(R_t, 0)` and :math:`R` is defined as for
    :func:`sharpe`.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices, performances or index.
    rf : float, optional
        Annualized risk-free rate. Default is 0.
    period : int, optional
        Number of periods per year. Default is 252 (trading days).
    log : bool, optional
        If True, compute returns as log-returns. Default is False.
    axis : {0, 1}, optional
        Axis along which the computation is done. Default is 0.
    dtype : np.dtype, optional
        Output array dtype. Inferred from ``X`` if not given.
    ddof : int, optional
        Delta Degrees of Freedom. Default is 0.

    Returns
    -------
    dtype or np.ndarray[dtype, ndim=1]
        Sortino ratio for each series. Returns ``inf`` when downside
        volatility is zero (all returns are non-negative).

    Examples
    --------
    Assume a series X of monthly prices:

    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> sortino(X, period=12)
    0.4742428587192754
    >>> sortino(X.reshape([6, 1]), period=12)
    array([0.47424286])

    See Also
    --------
    sharpe, calmar, mdd

    """
    ret = _annual_return(X, period, ddof)
    downside_vol = _annual_downside_volatility(X, period, log, axis, ddof)

    if (downside_vol == 0.).any():
        res = ret - rf
        res[downside_vol == 0.] = np.inf
        res[downside_vol != 0.] = res[downside_vol != 0.] / downside_vol[downside_vol != 0.]

        return res

    return (ret - rf) / downside_vol


@WrapperArray('axis')
def directional_accuracy(
    y_true: NDArray, y_pred: NDArray, axis: int = 0,
) -> float:
    r""" Compute the directional accuracy of a prediction.

    Fraction of periods where the predicted direction (sign) matches the
    true direction. A value of 1.0 means perfect directional alignment;
    0.5 is random; 0.0 means systematically wrong direction.

    Notes
    -----
    .. math::

        directionalAccuracy = \frac{1}{T} \sum_{t=1}^{T}
            \mathbf{1}[\text{sign}(\hat{y}_t) = \text{sign}(y_t)]

    Parameters
    ----------
    y_true : np.ndarray[ndim=1 or 2, dtype]
        Vector of true values (returns or price changes).
    y_pred : np.ndarray[ndim=1 or 2, dtype]
        Vector of predicted values.
    axis : {0, 1}, optional
        Axis along which the computation is done. Default is 0.

    Returns
    -------
    float or np.ndarray[ndim=1, float]
        Directional accuracy between 0 and 1.

    Examples
    --------
    >>> y_true = np.array([1., .5, -.5, .8, -.2])
    >>> y_pred = np.array([.5, .2, -.5, .1, .0])
    >>> directional_accuracy(y_true, y_pred)
    0.8

    See Also
    --------
    accuracy

    """
    return np.mean(np.sign(y_true) == np.sign(y_pred), axis=axis)


@WrapperArray('dtype', 'axis', 'window')
def z_score(X: NDArray, w: int = 0, kind: str = 's', axis: int = 0, dtype=None) -> NDArray:
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
          :func:`~fynance.features.momentums.ema` for details.
        - If 's' (default) then use simple moving average, see
          :func:`~fynance.features.momentums.sma` for details.
        - If 'w' then use weighted moving average, see
          :func:`~fynance.features.momentums.wma` for details.
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
    -1.224744871391589
    >>> z_score(X.reshape([6, 1]), w=3)
    array([-1.22474487])

    See Also
    --------
    roll_z_score, mdd, calmar, drawdown, sharpe

    """
    # TODO : make a more efficient function
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


@WrapperArray('dtype', 'axis', 'window', 'ddof', min_size=2)
def roll_annual_return(X: NDArray, period: int = 252, w: int | None = None, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute rolling compouned annual returns of each `X`' series.

    The annualised return [1]_ is the process of converting returns on a whole
    period to returns per year.

    Notes
    -----
    The rolling annual compouned returns is computed such that :math:`\forall t
    \in [1: T]`:

    .. math::

        annualReturn_t = \frac{X_t}{X_1}^{\frac{period}{t}} - 1

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of price, performance or index.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    w : int, optional
        Size of the lagged window of the rolling function, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``t - ddof``, where ``t`` represents the number of elements in time
        axis. Default is 0.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Values of rolling compouned annual returns of each series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rate_of_return#Annualisation

    Examples
    --------
    Assume series of monthly prices:

    >>> X = np.array([100, 110, 80, 120, 160, 108]).astype(np.float64)
    >>> roll_annual_return(X, period=12)
    array([ 0.        ,  0.771561  , -0.5904    ,  0.728     ,  2.08949828,
            0.1664    ])
    >>> X = np.array([[100, 101], [80, 81], [110, 108]]).astype(np.float64)
    >>> roll_annual_return(X, period=12, axis=1)
    array([[ 0.        ,  0.06152015],
           [ 0.        ,  0.07738318],
           [ 0.        , -0.10425081]])

    See Also
    --------
    mdd, drawdown, sharpe, annual_volatility

    """
    return _roll_annual_return(X, period, w, ddof)


def _roll_annual_return(X, period, w, ddof):
    if ddof >= w:

        raise ValueError(
            'size of the lagged window (w={}) must be strictly greater than '
            'degree of freedom (ddof={})'.format(w, ddof)
        )

    elif (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    elif len(X.shape) == 2:

        return np.asarray(roll_annual_return_cy_2d(X, period, w, ddof))

    return np.asarray(roll_annual_return_cy_1d(X, period, w, ddof))


@WrapperArray('dtype', 'axis', 'null', 'window', 'ddof', min_size=3)
def roll_annual_volatility(
    X: NDArray, period: int = 252, log: bool = True, w: int | None = None,
    axis: int = 0, dtype=None, ddof: int = 0,
) -> NDArray:
    r""" Compute the annualized volatility of each `X`' series.

    In finance, volatility is the degree of variation of a trading price
    series over time as measured by the standard deviation of logarithmic
    returns [2]_.

    Notes
    -----
    The rolling annualized volatility of returns is computed such that
    :math:`\forall t \in [1, T]`:

    .. math::

        annualVolatility_t = \sqrt{period \times Var(R_{1:t})} \\ \\

    Where, :math:`R_1 = 0` and :math:`R_{2:t} = \begin{cases}ln(\frac{X_{2:t}}
    {X_{1:t-1}}) \text{, if log=True}\\ \frac{X_{2:t}}{X_{1:t-1}} - 1 \text{,
    otherwise} \\ \end{cases}`.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of price, performance or index.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    log : bool, optional
        - If True then logarithmic returns are computed.
        - Else then returns in percentage are computed.
    w : int, optional
        Size of the lagged window of the rolling function, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``t - ddof``, where ``t`` represents the number of elements in time
        axis. Default is 0.

    Returns
    -------
    dtype or np.ndarray([dtype, ndim=1 or 2])
        Rolling annualized volatility for each series.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Volatility_(finance)

    Examples
    --------
    Assume series of monthly prices:

    >>> X = np.array([100, 110, 105, 110, 120, 108]).astype(np.float64)
    >>> roll_annual_volatility(X, period=12, log=False, ddof=1)
    array([0.        , 0.24494897, 0.25777176, 0.21655755, 0.21313847,
           0.27344193])
    >>> roll_annual_volatility(X.reshape([6, 1]), period=12, log=False)
    array([[0.        ],
           [0.17320508],
           [0.21046976],
           [0.18754434],
           [0.19063685],
           [0.24961719]])

    See Also
    --------
    mdd, drawdown, sharpe, annual_return

    """
    return _roll_annual_volatility(X, period, log, w, axis, ddof)


def _roll_annual_volatility(X, period, log, w, axis, ddof):
    if ddof >= w:

        raise ValueError(
            'size of the lagged window (w={}) must be strictly greater than '
            'degree of freedom (ddof={})'.format(w, ddof)
        )

    elif len(X.shape) == 2:

        return np.asarray(roll_annual_volatility_cy_2d(
            X, period, int(log), w, ddof
        ))

    return np.asarray(roll_annual_volatility_cy_1d(
        X, period, int(log), w, ddof
    ))


@WrapperArray('dtype', 'axis', 'window', 'ddof', min_size=2)
def roll_calmar(X: NDArray, period: float = 252., w: int | None = None, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute the rolling Calmar ratio of each `X`' series.

    Notes
    -----
    Calmar ratio [3]_ is the rolling compouned annual return
    (:func:`~fynance.features.metrics.roll_annual_return`) over the rolling
    maximum drawdown (:func:`~fynance.features.metrics.roll_mdd`). Let
    :math:`T` the number of time observations, DD the vector of drawdown,
    :math:`\forall t \in [1:T]`:

    .. math::

        calmarRatio_t = \frac{annualReturn_t}{MDD_t} \\ \\

    With, :math:`annualReturn_t = \frac{X_t}{X_1}^{\frac{period}{t}} - 1` and
    :math:`MDD_t = max(DD_t)`, where
    :math:`DD_t = 1 - \frac{X_t}{max(X_{1:t})}`.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of price, performance or index.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    w : int, optional
        Size of the lagged window of the rolling function, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``t - ddof``, where ``t`` represents the number of elements in time
        axis. Default is 0.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Series of rolling Calmar ratio.

    References
    ----------
    .. [3] https://en.wikipedia.org/wiki/Calmar_ratio

    Examples
    --------
    Assume a monthly series of prices:

    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> roll_calmar(X, period=12)
    array([ 0.        ,  0.        ,  3.52977926, 20.18950437, 31.35989887,
            0.6122449 ])

    See Also
    --------
    roll_mdd, roll_sharpe, calmar

    """
    ret = _roll_annual_return(X, period, w, ddof)
    mdd = _roll_mdd(X, w, False)
    calmar = np.zeros(X.shape)
    slice_bool = (mdd != 0)
    calmar[slice_bool] = ret[slice_bool] / mdd[slice_bool]

    return calmar


@WrapperArray('dtype', 'axis', 'window')
def roll_drawdown(X: NDArray, w: int | None = None, raw: bool = False, axis: int = 0, dtype=None) -> NDArray:
    r""" Measures the rolling drawdown of each `X`' series.

    Function to compute measure of the decline from a historical peak in some
    variable [5]_ (typically the cumulative profit or total open equity of a
    financial trading strategy).

    Notes
    -----
    Let DD^w the drawdown vector with a lagged window of size `w`:

    .. math::

        DD^w_t =\begin{cases}
        max(X_{t - w + 1:t}) - X_t \text{, if raw=True} \\
        1 - \frac{X_t}{max(X_{t - w + 1:t})} \text{, otherwise} \\
        \end{cases}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices, performances or index. Must be positive values.
    w : int, optional
        Size of the lagged window of the rolling function, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
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
    >>> roll_drawdown(X)
    array([0. , 0. , 0.2, 0. , 0. , 0.5])
    >>> roll_drawdown(X.reshape([6, 1])).T
    array([[0. , 0. , 0.2, 0. , 0. , 0.5]])
    >>> roll_drawdown(X, raw=True)
    array([ 0.,  0., 20.,  0.,  0., 80.])
    >>> X = np.array([100, 80, 70, 75, 110, 80]).astype(np.float64)
    >>> roll_drawdown(X, raw=True, w=3)
    array([ 0., 20., 30.,  5.,  0., 30.])

    See Also
    --------
    mdd, calmar, sharpe, roll_mdd

    """
    return _roll_drawdown(X, w, raw)


def _roll_drawdown(X, w, raw):
    if (X[0] == 0).any() and not raw:

        warn(
            'Cannot compute drawdown in percentage without initial values '
            'X[0] strictly positive.',
            category=UserWarning,
            stacklevel=2,
        )
        raw = True

    if len(X.shape) == 2:

        return np.asarray(roll_drawdown_cy_2d(X, w, int(raw)))

    return np.asarray(roll_drawdown_cy_1d(X, w, int(raw)))


@WrapperArray('dtype', 'axis', 'window')
def roll_mad(X: NDArray, w: int | None = None, axis: int = 0, dtype=None) -> NDArray:
    """ Compute rolling Mean Absolut Deviation for each `X`' series.

    Compute the moving average of the absolute value of the distance to the
    moving average [6]_.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time series (price, performance or index).
    w : int, optional
        Size of the lagged window of the rolling function, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
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
    .. [6] https://en.wikipedia.org/wiki/Average_absolute_deviation

    Examples
    --------
    >>> X = np.array([70, 100, 90, 110, 150, 80])
    >>> roll_mad(X, dtype=np.float64)
    array([ 0.        , 15.        , 11.11111111, 12.5       , 20.8       ,
           20.        ])
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> roll_mad(X, w=3, dtype=np.float64)
    array([ 0.        , 20.        , 13.33333333, 13.33333333, 26.66666667,
           26.66666667])

    See Also
    --------
    mad

    """
    if len(X.shape) == 2:

        return np.asarray(roll_mad_cy_2d(X, w))

    return np.asarray(roll_mad_cy_1d(X, w))


@WrapperArray('dtype', 'axis', 'window')
def roll_mdd(X: NDArray, w: int | None = None, raw: bool = False, axis: int = 0, dtype=None) -> NDArray:
    """ Compute the rolling maximum drawdown for each `X`' series.

    Where drawdown is the measure of the decline from a historical peak in
    some variable [5]_ (typically the cumulative profit or total open equity
    of a financial trading strategy).

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time series (price, performance or index).
    w : int, optional
        Size of the lagged window of the rolling function, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
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
    np.ndrray[dtype, ndim=1 or 2]
        Series of rolling maximum drawdown for each series.

    References
    ----------
    .. [5] https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

    Examples
    --------
    >>> X = np.array([70, 100, 80, 120, 160, 80])
    >>> roll_mdd(X, dtype=np.float64)
    array([0. , 0. , 0.2, 0.2, 0.2, 0.5])
    >>> roll_mdd(X, w=3, dtype=np.float64)
    array([0. , 0. , 0.2, 0.2, 0. , 0.5])
    >>> X = np.array([100, 80, 70, 75, 110, 80]).astype(np.float64)
    >>> roll_mdd(X, raw=True, w=3, dtype=np.float64)
    array([ 0., 20., 30., 10.,  0., 30.])

    See Also
    --------
    mdd, roll_calmar, roll_sharpe, drawdown

    """
    return _roll_mdd(X, w, raw)


def _roll_mdd(X, w, raw):
    if len(X.shape) == 2:

        return np.asarray(roll_mdd_cy_2d(X, w, int(raw)))

    return np.asarray(roll_mdd_cy_1d(X, w, int(raw)))


@WrapperArray('dtype', 'axis', 'window', 'ddof', min_size=2)
def roll_sharpe(
    X: NDArray, rf: float = 0, period: int = 252, w: int | None = None,
    log: bool = False, axis: int = 0, dtype=None, ddof: int = 0,
) -> NDArray:
    r""" Compute rolling sharpe ratio of each `X`' series.

    Notes
    -----
    Sharpe ratio [7]_ is computed as the rolling annualized expected returns
    (:func:`~roll_annual_return`) minus the risk-free rate (noted :math:`rf`)
    over the rolling annualized volatility of returns
    (:func:`~roll_annual_volatility`) such that :math:`\forall t \in [1:T]`:

    .. math::

        sharpeRatio_t = \frac{E(R | R_{1:t}) - rf_t}{\sqrt{period \times
        Var(R | R_{1:t})}}

    Where, :math:`R_1 = 0` and :math:`R_{2:T} = \begin{cases}ln(\frac{X_{2:T}}
    {X_{1:T-1}}) \text{, if log=True}\\ \frac{X_{2:T}}{X_{1:T-1}} - 1 \text{,
    otherwise} \\ \end{cases}`.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices, performances or index.
    rf : float or np.ndarray[dtype, ndim=1 or 2], optional
        Means the annualized risk-free rate, default is 0. If an array is
        passed, it must be of the same shape than ``X``.
    period : int, optional
        Number of period per year, default is 252 (trading days).
    w : int, optional
        Size of the lagged window of the rolling function, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    log : bool, optional
        If true compute sharpe with the formula for log-returns, default
        is False.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.
    ddof : int, optional
        Means Delta Degrees of Freedom, the divisor used in calculations is
        ``T - ddof``, where ``T`` represents the number of elements in time
        axis. Default is 0.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Serires of rolling Sharpe ratio.

    References
    ----------
    .. [7] https://en.wikipedia.org/wiki/Sharpe_ratio

    Examples
    --------
    Assume a monthly series of prices:

    >>> X = np.array([70, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> roll_sharpe(X, period=12)
    array([ 0.        , 10.10344078,  0.77721579,  3.99243019,  6.754557  ,
            0.24475518])

    See Also
    --------
    roll_calmar, sharpe, roll_mdd

    """
    ret = _roll_annual_return(X, period, w, ddof)
    vol = _roll_annual_volatility(X, period, log, w, axis, ddof)
    sharpe = np.zeros(X.shape)
    slice_bool = (vol != 0)

    if isinstance(rf, float) or isinstance(rf, int):
        _rf = rf

    elif isinstance(rf, np.ndarray) and rf.shape[0] != X.shape[0]:
        msg_prefix = 'rf must be '

        raise ArraySizeError(X.shape[0], msg_prefix=msg_prefix)

    else:
        _rf = rf[slice_bool]

    sharpe[slice_bool] = (ret[slice_bool] - _rf) / vol[slice_bool]

    return sharpe


@WrapperArray('dtype', 'axis', 'window')
def roll_z_score(X: NDArray, w: int | None = None, kind: str = 's', axis: int = 0, dtype=None) -> NDArray:
    r""" Compute vector of rolling/moving Z-score function.

    Notes
    -----
    Compute for each observation the z-score function for a specific moving
    average function such that :math:`\forall t \in [1:T]`:

    .. math::

        z_t = \frac{X_t - \mu_t}{\sigma_t}

    Where :math:`\mu_t` is the moving average and :math:`\sigma_t` is the
    moving standard deviation.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Series of index, prices or returns.
    w : int, optional
        Size of the lagged window of the moving averages, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    kind : {'e', 's', 'w'}
        - If 'e' then use exponential moving average, see
          :func:`~fynance.features.momentums.ema` for details.
        - If 's' (default) then use simple moving average, see
          :func:`~fynance.features.momentums.sma` for details.
        - If 'w' then use weighted moving average, see
          :func:`~fynance.features.momentums.wma` for details.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
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
    if kind == 'e':
        w = 1 - 2 / (1 + w)

    avg = _handler_ma[kind.lower()](X, w)
    std = _handler_mstd[kind.lower()](X, w)

    std[std == 0.] = 1.
    z = (X - avg) / std

    return z


# =========================================================================== #
#                                 old scripts                                 #
# =========================================================================== #


def _roll_annual_return_py(X, period, w, ddof):
    """ Old function. """
    if (X[0] == 0).any():

        raise ValueError('initial value X[0] cannot be null.')

    cum_ret = np.zeros(X.shape)
    cum_ret[: w] = X[: w] / X[0]
    cum_ret[w:] = X[w:] / X[: -w]

    if (cum_ret < 0).any():

        raise ValueError('all values of X must be of the same sign.')

    T = X.shape[0]
    power = period / np.arange(1, T - ddof + 1, dtype=np.float64)

    if len(X.shape) == 2:
        power = power.reshape([T, 1])

    sign = np.sign(X[0])

    anu_ret = np.zeros(X.shape)
    anu_ret[ddof:] = sign * np.float_power(cum_ret[ddof:], power) - 1.

    return anu_ret


def _roll_annual_volatility_py(X, period, log, w, axis, ddof):
    """ Old function. """
    shape = X.shape
    T = shape[0]
    R = np.zeros(shape)
    anu_vol = np.zeros(shape)

    if log:
        R[1:] = np.log(X[1:] / X[:-1])

    else:
        R[1:] = X[1:] / X[:-1] - 1.

    for t in range(ddof + 1, T):
        t0 = max(0, t - w)
        anu_vol[t] = np.std(R[t0:t + 1], axis=axis, ddof=ddof)

    return np.sqrt(period) * anu_vol


# =========================================================================== #
#                                   Tests                                     #
# =========================================================================== #


if __name__ == '__main__':

    import doctest

    doctest.testmod()
