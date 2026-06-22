#!/usr/bin/env python3
# coding: utf-8

""" Statistical helpers (accuracy, directional accuracy, z-score). """

from __future__ import annotations

# Built-in packages
# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance._wrappers import WrapperArray
from fynance.features._metrics_helpers import *  # noqa: F401,F403

__all__ = ['accuracy', 'directional_accuracy', 'percent_positive',
           'tail_ratio', 'z_score', 'roll_z_score', 'mad', 'roll_mad']


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
    # No `wrap_axis` here: it can only transpose the first positional argument,
    # leaving `y_pred` mis-oriented for axis=1. Both arrays already share the
    # same orientation, so reduce directly along `axis` (NumPy handles the
    # axis bounds, including negative axes).
    if sign:
        y_true = np.sign(y_true)
        y_pred = np.sign(y_pred)

    return np.sum(y_true == y_pred, axis=axis) / y_true.shape[axis]


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
    # No `wrap_axis` here: it can only transpose the first positional argument,
    # leaving `y_pred` mis-oriented for axis=1. Both arrays already share the
    # same orientation, so reduce directly along `axis` (NumPy handles the
    # axis bounds, including negative axes).
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
    if kind == 'e':
        w = 1 - 2 / (1 + w)  # type: ignore[assignment]

    avg = _handler_ma[kind.lower()](X, w)
    std = _handler_mstd[kind.lower()](X, w)

    std[std == 0.] = 1.
    z = (X - avg) / std

    return z[-1]


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
        w = 1 - 2 / (1 + w)  # type: ignore[assignment, operator]

    avg = _handler_ma[kind.lower()](X, w)
    std = _handler_mstd[kind.lower()](X, w)

    std[std == 0.] = 1.
    z = (X - avg) / std

    return z


@WrapperArray('axis')
def percent_positive(X: NDArray, axis: int = 0) -> NDArray:
    r""" Fraction of strictly positive observations.

    A simple robustness statistic: the share of periods with a positive
    return (the "hit rate" / percentage of winning periods).

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Series of returns.
    axis : {0, 1}, optional
        Axis of computation. Default 0.

    Returns
    -------
    float or np.ndarray
        Fraction in ``[0, 1]`` of strictly positive values.

    Examples
    --------
    >>> X = np.array([0.1, -0.2, 0.3, 0.0, 0.4])
    >>> float(percent_positive(X))
    0.6

    """
    return np.mean(X > 0, axis=0)


@WrapperArray('axis')
def tail_ratio(X: NDArray, alpha: float = 0.05, axis: int = 0) -> NDArray:
    r""" Tail ratio of a return series.

    Ratio of the magnitude of the right tail to the left tail:
    :math:`|q_{1-\alpha}| / |q_{\alpha}|`. A value above 1 means the
    upside tail is larger than the downside tail.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Series of returns.
    alpha : float, optional
        Tail quantile level. Default 0.05 (95th vs 5th percentile).
    axis : {0, 1}, optional
        Axis of computation. Default 0.

    Returns
    -------
    float or np.ndarray
        Tail ratio (0 when the left tail is exactly 0).

    """
    hi = np.abs(np.quantile(X, 1.0 - alpha, axis=0))
    lo = np.abs(np.quantile(X, alpha, axis=0))

    return np.where(lo > 0, hi / np.where(lo > 0, lo, 1.0), 0.0)


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
    return np.mean(np.abs(X - np.mean(X, axis=axis, keepdims=True)), axis=axis)


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

        return _roll_mad_2d(X, w)

    return _roll_mad_1d(X, w)
