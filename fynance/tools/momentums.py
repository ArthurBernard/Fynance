#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-20 19:57:13
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-16 20:35:32

""" Statical momentum functions. """

# Built-in packages

# External packages
import numpy as np

# Local packages
from fynance.tools.momentums_cy import *
from fynance.tools._wrappers import WrapperArray

__all__ = [
    'sma', 'wma', 'ema', 'smstd', 'wmstd', 'emstd',
]

# TODO : - Momentums of order 3
#        - Momentums of order 4
#        - Momentums of order w

# =========================================================================== #
#                               Moving Averages                               #
# =========================================================================== #


@WrapperArray('dtype', 'axis', 'window')
def sma(X, w=None, axis=0, dtype=None):
    r""" Compute simple moving average(s) of size `w` for each `X`' series.

    .. math::

        sma^w(X_t) = \frac{1}{w} \sum^{w-1}_{i=0} X_{t-i}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the moving average.
    w : int, optional
        Size of the lagged window of the moving average, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Simple moving average of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> sma(X, w=3, dtype=np.float64, axis=0)
    array([ 60.,  80.,  80., 100., 120., 120.])
    >>> X = np.array([[60, 60], [100, 100], [80, 80],
    ...               [120, 120], [160, 160], [80, 80]])
    >>> sma(X, w=3, dtype=np.float64, axis=0)
    array([[ 60.,  60.],
           [ 80.,  80.],
           [ 80.,  80.],
           [100., 100.],
           [120., 120.],
           [120., 120.]])
    >>> sma(X, w=3, dtype=np.float64, axis=1)
    array([[ 60.,  60.],
           [100., 100.],
           [ 80.,  80.],
           [120., 120.],
           [160., 160.],
           [ 80.,  80.]])


    See Also
    --------
    wma, ema, smstd

    """
    return _sma(X, w, dtype=dtype, axis=axis)


def _sma(X, w, axis=0, dtype=None):
    if len(X.shape) == 2:

        return np.asarray(sma_cy_2d(X, w))

    return np.asarray(sma_cy_1d(X, w))


@WrapperArray('dtype', 'axis', 'window')
def wma(X, w=None, axis=0, dtype=None):
    r""" Compute weighted moving average(s) of size `w` for each `X`' series.

    .. math::

        wma^w(X_t) = \frac{2}{w (w-1)} \sum^{w-1}_{i=0} (w-i) \times X_{t-i}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the moving average.
    w : int, optional
        Size of the lagged window of the moving average, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Weighted moving average of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> wma(X, w=3, dtype=np.float64)
    array([ 60.        ,  86.66666667,  83.33333333, 103.33333333,
           133.33333333, 113.33333333])
    >>> X = X.reshape([6, 1])
    >>> wma(X, w=3, dtype=np.float64).flatten()
    array([ 60.        ,  86.66666667,  83.33333333, 103.33333333,
           133.33333333, 113.33333333])

    See Also
    --------
    sma, ema, wmstd

    """
    return _wma(X, w, axis=axis, dtype=dtype)


def _wma(X, w, axis=0, dtype=None):
    if len(X.shape) == 2:

        return np.asarray(wma_cy_2d(X, w))

    return np.asarray(wma_cy_1d(X, w))


@WrapperArray('dtype', 'axis')
def ema(X, alpha=0.94, w=None, axis=0, dtype=None):
    r""" Compute exponential moving average(s) for each `X`' series.

    .. math::

        ema^{\apha}(X_t) = \alpha \times ema_{t-1} + (1-\alpha) \times X_t

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the moving average.
    alpha : float, optional
        These coefficient represents the degree of weighting decrease, default
        is 0.94 corresponding at 20 lags memory.
    w : int, optional
        Size of the lagged window of the moving average, must be strictly
        positive. If ``w is None`` the window is ignored and the parameter
        `alpha` is used. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Exponential moving average of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> ema(X, w=3, dtype=np.float64)
    array([ 60.,  80.,  80., 100., 130., 105.])
    >>> ema(X, alpha=0.5, dtype=np.float64)
    array([ 60.,  80.,  80., 100., 130., 105.])

    Notes
    -----
    If the lagged window `w` is specified :math:`\alpha` is overwritten by
    :math:`\alpha = 1 - \frac{2}{1 + w}`

    See Also
    --------
    sma, wma, emstd

    """
    if w is None:
        pass

    elif w <= 0:

        raise ValueError('lagged window of size {} is not available, \
            must be greater than 0.'.format(w))

    else:
        alpha = 1 - 2 / (1 + w)

    return _ema(X, alpha, axis=axis, dtype=dtype)


def _ema(X, alpha, axis=0, dtype=None):
    if len(X.shape) == 2:

        return np.asarray(ema_cy_2d(X, float(alpha)))

    return np.asarray(ema_cy_1d(X, float(alpha)))


# =========================================================================== #
#                          Moving Standard Deviation                          #
# =========================================================================== #


@WrapperArray('dtype', 'axis', 'window')
def smstd(X, w=None, axis=0, dtype=None):
    r""" Compute simple moving standard deviation(s) for each `X`' series'.

    .. math::

        smstd^w(X_t) = \sqrt{\frac{1}{w} \sum^{w-1}_{i=0} (X_{t-i} - sma_t)^2}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the moving standard deviation.
    w : int, optional
        Size of the lagged window of the moving average, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Simple moving standard deviation of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> smstd(X, w=3, dtype=np.float64)
    array([ 0.        , 20.        , 16.32993162, 16.32993162, 32.65986324,
           32.65986324])
    >>> smstd(X.reshape([6, 1]), w=3, dtype=np.float64).flatten()
    array([ 0.        , 20.        , 16.32993162, 16.32993162, 32.65986324,
           32.65986324])

    See Also
    --------
    sma, wmstd, emstd

    """
    return _smstd(X, w, axis=axis, dtype=dtype)


def _smstd(X, w, axis=0, dtype=None):
    if len(X.shape) == 2:

        return np.asarray(smstd_cy_2d(X, w))

    return np.asarray(smstd_cy_1d(X, w))


@WrapperArray('dtype', 'axis', 'window')
def wmstd(X, w=None, axis=0, dtype=None):
    r""" Compute weighted moving standard(s) deviation for each `X`' series'.

    .. math::

        wma_t = \frac{2}{w (w-1)} \sum^{w-1}_{i=0} (w-i) \times X_{t-i} \\
        wmstd^w(X_t) = \sqrt{\frac{2}{w(w-1)} \sum^{w-1}_{i=0}
        (w-i) \times (X_{t-i} - wma_t)^2}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the moving standard deviation.
    w : int, optional
        Size of the lagged window of the moving average, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Weighted moving standard deviation of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> wmstd(X, w=3, dtype=np.float64)
    array([ 0.        , 18.85618083, 13.74368542, 17.95054936, 29.8142397 ,
           35.90109871])
    >>> wmstd(X.reshape([6, 1]), w=3, dtype=np.float64).flatten()
    array([ 0.        , 18.85618083, 13.74368542, 17.95054936, 29.8142397 ,
           35.90109871])

    See Also
    --------
    wma, smstd, emstd

    """
    return _wmstd(X, w, axis=axis, dtype=dtype)


def _wmstd(X, w, axis=0, dtype=None):
    if len(X.shape) == 2:

        return np.asarray(wmstd_cy_2d(X, w))

    return np.asarray(wmstd_cy_1d(X, w))


@WrapperArray('dtype', 'axis')
def emstd(X, alpha=0.94, w=None, axis=0, dtype=None):
    r""" Compute exponential moving standard deviation(s) for each `X`' series.

    .. math::

        emstd^{\alpha}(X_t) = \sqrt{\alpha\times emstd_{t-1}^2+(1-\alpha)
        \times X_t^2}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the moving standard deviation.
    alpha : float, optional
        These coefficient represents the degree of weighting decrease, default
        is 0.94 corresponding at 20 lags memory.
    w : int, optional
        Size of the lagged window of the moving average, must be strictly
        positive. If ``w is None`` the window is ignored and the parameter
        `alpha` is used. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Exponential moving standard deviation of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> emstd(X, w=3, dtype=np.float64)
    array([ 0.        , 14.14213562, 10.        , 15.8113883 , 23.97915762,
           24.49489743])
    >>> emstd(X.reshape([6, 1]), w=3, dtype=np.float64).flatten()
    array([ 0.        , 14.14213562, 10.        , 15.8113883 , 23.97915762,
           24.49489743])
    >>> emstd(X, alpha=0.5, dtype=np.float64)
    array([ 0.        , 14.14213562, 10.        , 15.8113883 , 23.97915762,
           24.49489743])

    Notes
    -----
    If the lagged window `w` is specified :math:`\alpha` is overwritten by
    :math:`\alpha = 1 - \frac{2}{1 + w}`

    See Also
    --------
    ema, smstd, wmstd

    """
    if w is None:
        pass

    elif w <= 0:

        raise ValueError('lagged window of size {} is not available, \
            must be greater than 0.'.format(w))

    else:
        alpha = 1 - 2 / (1 + w)

    return _emstd(X, alpha, axis=axis, dtype=dtype)


def _emstd(X, alpha, axis=0, dtype=None):
    if len(X.shape) == 2:

        return np.asarray(emstd_cy_2d(X, float(alpha)))

    return np.asarray(emstd_cy_1d(X, float(alpha)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
