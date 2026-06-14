#!/usr/bin/env python3
# coding: utf-8

""" Drawdown statistics (drawdown path, max drawdown, mean abs deviation). """

from __future__ import annotations

# Built-in packages
# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance._wrappers import WrapperArray
from fynance.features._metrics_helpers import *  # noqa: F401,F403
from fynance.features.metrics_cy import *

__all__ = ['drawdown', 'mdd', 'mad', 'roll_drawdown', 'roll_mdd', 'roll_mad']


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

