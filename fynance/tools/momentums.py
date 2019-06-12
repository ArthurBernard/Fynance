#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-20 19:57:13
# @Last modified by: ArthurBernard
# @Last modified time: 2019-06-12 15:41:57

# Built-in packages

# External packages
import numpy as np

# Local packages
from .momentums_cy import *

__all__ = [
    'sma', 'wma', 'ema', 'smstd', 'wmstd', 'emstd',
]

# TODO : - Momentums of order 3
#        - Momentums of order 4
#        - Momentums of order k

# =========================================================================== #
#                               Moving Averages                               #
# =========================================================================== #


def sma(series, lags=21):
    r""" Simple moving average along k lags.

    .. math:: sma_t = \frac{1}{k} \sum^{k-1}_{i=0} series_{t-i}

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    lags : int, optional
        Number of lags for ma, default is 21.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Vector of moving average of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> sma(series, lags=3)
    array([ 60.,  80.,  80., 100., 120., 120.])

    See Also
    --------
    wma, ema

    """
    return sma_cy(series.flatten().astype(np.float64), lags=int(lags))


def wma(series, lags=21):
    r""" Weighted moving average along k lags.

    .. math::
        wma_t = \frac{2}{k (k-1)} \sum^{k-1}_{i=0} (k-i) \times series_{t-i}

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    lags : int, optional
        Number of lags for ma, default is 21.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving average of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> wma(series, lags=3)
    array([ 60.        ,  86.66666667,  83.33333333, 103.33333333,
           133.33333333, 113.33333333])

    See Also
    --------
    sma, ema

    """
    return wma_cy(series.flatten().astype(np.float64), lags=int(lags))


def ema(series, alpha=0.94, lags=None):
    r""" Exponential moving average along k lags.

    .. math::
        ema_t = \alpha \times ema_{t-1} + (1-\alpha) \times series_t

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    alpha : float, optional
        Multiplier, default is 0.94 corresponding at 20 lags memory.
    lags : int, optional
        Number of days.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving average of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> ema(series, lags=3)
    array([ 60.,  80.,  80., 100., 130., 105.])
    >>> ema(series, alpha=0.5)
    array([ 60.,  80.,  80., 100., 130., 105.])

    Notes
    -----
    If `lags` is specified :math:`\alpha = 1 - \frac{2}{1 + k}`

    See Also
    --------
    sma, wma

    """
    if lags is not None:
        alpha = 1 - 2 / (1 + lags)

    return ema_cy(series.flatten().astype(np.float64), alpha=float(alpha))


# =========================================================================== #
#                          Moving Standard Deviation                          #
# =========================================================================== #


def smstd(series, lags=21):
    r""" Simple moving standard deviation along k lags.

    .. math::
        smstd_t = \sqrt{\frac{1}{k} \sum^{k-1}_{i=0} (p_{t-i} - sma_t)^2}

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    lags : int, optional
        Number of lags for ma, default is 21.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> smstd(series, lags=3)
    array([ 0.        , 20.        , 16.32993162, 16.32993162, 32.65986324,
           32.65986324])

    See Also
    --------
    wmstd, emstd

    """
    return smstd_cy(series.flatten().astype(np.float64), lags=int(lags))


def wmstd(series, lags=21):
    r""" Weighted moving standard deviation along k lags.

    .. math::
        wma_t = \frac{2}{k (k-1)} \sum^{k-1}_{i=0} (k-i) \times series_{t-i}
        wmstd_t = \sqrt{\frac{2}{k(k-1)}\sum^{k-1}_{i=0}(k-i)\times (series_{t-i}-wma_t)^2}

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Series of prices, index or returns.
    lags : int, optional
        Number of days, default is 21.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> wmstd(series, lags=3)
    array([ 0.        , 18.85618083, 13.74368542, 17.95054936, 29.8142397 ,
           35.90109871])

    See Also
    --------
    smstd, emstd

    """
    return wmstd_cy(series.flatten().astype(np.float64), lags=int(lags))


def emstd(series, alpha=0.94, lags=None):
    r""" Exponential moving standard deviation along k lags.

    .. math::
        emstd_t = \sqrt{\alpha\times emstd_{t-1}^2+(1-\alpha)\times series_t^2}

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Index or returns.
    alpha : float, optional
        Multiplier, default is 0.94 corresponding at 20 lags memory.
    lags : int, optional
        Number of days.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> emstd(series, lags=3)
    array([ 0.        , 28.28427125, 20.        , 31.6227766 , 47.95831523,
           48.98979486])

    Notes
    -----
    If `lags` is specified :math:`\alpha = 1 - \frac{2}{1 + k}`

    See Also
    --------
    smstd, wmstd

    """
    if lags is not None:
        alpha = 1 - 2 / (1 + lags)

    return emstd_cy(series.flatten().astype(np.float64), alpha=float(alpha))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
