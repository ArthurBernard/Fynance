#!/usr/bin/env python3
# coding: utf-8

""" Return and performance metrics (annualized return, P&L paths). """

from __future__ import annotations

# Built-in packages
# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance._wrappers import WrapperArray
from fynance.features._metrics_helpers import *  # noqa: F401,F403

__all__ = ['annual_return', 'perf_index', 'perf_returns', 'roll_annual_return']


@WrapperArray('dtype', 'axis', 'ddof', min_size=2)
def annual_return(X: NDArray, period: int = 252, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute compounded annual returns of each `X`' series.

    The annualised return [1]_ is the process of converting returns on a whole
    period to returns per year.

    Notes
    -----
    Let T the number of timeframes in `X`' series, the annual compounded returns
    is computed such that:

    .. math::

        annualReturn = \left(\frac{X_T}{X_1}\right)^{\frac{period}{T}} - 1

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Time-series of price, performance or index.
    period : int, optional
        Number of period per year, default is 252 (trading days per year).
    axis : {0, 1}, optional
        Axis along which the computation is done. Default is 0.
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
        Values of compounded annual returns of each series.

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
        X = base * np.cumprod(R + 1., axis=axis)  # type: ignore[assignment]

    else:

        raise ValueError(
            f"unknown kind {kind!r} of returns, only 'raw', 'log', 'pct' "
            "are supported"
        )

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


@WrapperArray('dtype', 'axis', 'window', 'ddof', min_size=2)
def roll_annual_return(X: NDArray, period: int = 252, w: int | None = None, axis: int = 0, dtype=None, ddof: int = 0) -> NDArray:
    r""" Compute rolling compounded annual returns of each `X`' series.

    The annualised return [1]_ is the process of converting returns on a whole
    period to returns per year.

    Notes
    -----
    The rolling annual compounded returns is computed such that :math:`\forall t
    \in [1: T]`:

    .. math::

        annualReturn_t = \left(\frac{X_t}{X_1}\right)^{\frac{period}{t}} - 1

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
        Axis along which the computation is done. Default is 0.
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
        Values of rolling compounded annual returns of each series.

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

