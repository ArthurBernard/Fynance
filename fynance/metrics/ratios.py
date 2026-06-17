#!/usr/bin/env python3
# coding: utf-8

""" Risk-adjusted ratios (Sharpe, Sortino, Calmar, diversification) and volatility. """

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance._exceptions import ArraySizeError
from fynance._wrappers import WrapperArray
from fynance.features._metrics_helpers import *  # noqa: F401,F403

__all__ = ['annual_volatility', 'sharpe', 'sortino', 'calmar', 'diversified_ratio', 'roll_annual_volatility', 'roll_sharpe', 'roll_calmar']


def _safe_ratio(excess: NDArray, denom: NDArray) -> Any:
    """ ``excess / denom`` robust to a zero denominator (scalar **or** array).

    A zero-volatility series has an undefined ratio: it is ``+inf`` when the
    excess return is non-zero (a riskless gain), and ``0`` when the excess is also
    zero (a flat, do-nothing curve). Returns a numpy scalar for 0-D input and an
    array otherwise — matching the surrounding ratio functions' shape contract.
    """
    excess = np.asarray(excess, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)

    if denom.ndim == 0:
        if denom != 0.:

            return np.float64(excess / denom)

        return np.float64(np.inf if excess != 0. else 0.)

    res = np.full(denom.shape, np.inf, dtype=np.float64)
    nz = denom != 0.
    res[nz] = excess[nz] / denom[nz]
    res[(~nz) & (excess == 0.)] = 0.

    return res


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

    return _safe_ratio(ret - rf, vol)


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
    R = _compute_returns(X, log)
    ret = _annual_return(X, period, ddof)
    downside_vol = np.sqrt(period) * np.std(np.where(R < 0, R, 0.), axis=axis, ddof=ddof)

    return _safe_ratio(ret - rf, downside_vol)


@WrapperArray('dtype', 'axis', 'ddof', min_size=2)
def calmar(X: NDArray, period: int = 252, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute the Calmar Ratio for each `X`' series.

    Notes
    -----
    Calmar ratio [3]_ is the compouned annual return
    (:func:`~fynance.metrics.annual_return`) over the maximum drawdown
    (:func:`~fynance.metrics.mdd`). Let :math:`T` the number of time
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
    #        append examples
    T, N = X.shape

    if W is None:
        W = np.ones([N, 1]) / N

    else:
        W = W.reshape([N, 1])

    sigma = np.std(X, axis=0).reshape([N, 1])
    V = np.cov(X, rowvar=False, bias=True).reshape([N, N])

    return (W.T @ sigma) / np.sqrt(W.T @ V @ W)


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


@WrapperArray('dtype', 'axis', 'window', 'ddof', min_size=2)
def roll_calmar(X: NDArray, period: float = 252., w: int | None = None, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute the rolling Calmar ratio of each `X`' series.

    Notes
    -----
    Calmar ratio [3]_ is the rolling compouned annual return
    (:func:`~fynance.metrics.roll_annual_return`) over the rolling
    maximum drawdown (:func:`~fynance.metrics.roll_mdd`). Let
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

