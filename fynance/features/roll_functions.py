#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-18 21:15:59
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-18 21:49:35

""" Rolling functions. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages
from fynance.features.roll_functions_cy import *
from fynance._wrappers import WrapperArray

__all__ = ["roll_min", "roll_max"]


# =========================================================================== #
#                                   Min Max                                   #
# =========================================================================== #


@WrapperArray('dtype', 'axis', 'window')
def roll_min(X, w=None, axis=0, dtype=None):
    r""" Compute simple rolling minimum of size `w` for each `X`' series.

    .. math::

        roll_min^w_t(X) = min(X_{t - w}, ..., X_t)

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the rolling minimum.
    w : int, optional
        Size of the lagged window of the rolling minimum, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Simple rolling minimum of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> roll_min(X, w=3, dtype=np.float64, axis=0)
    array([60., 60., 60., 80., 80., 80.])
    >>> X = np.array([[60, 60], [100, 100], [80, 80],
    ...               [120, 120], [160, 160], [80, 80]])
    >>> roll_min(X, w=3, dtype=np.float64, axis=0)
    array([[60., 60.],
           [60., 60.],
           [60., 60.],
           [80., 80.],
           [80., 80.],
           [80., 80.]])
    >>> roll_min(X, w=3, dtype=np.float64, axis=1)
    array([[ 60.,  60.],
           [100., 100.],
           [ 80.,  80.],
           [120., 120.],
           [160., 160.],
           [ 80.,  80.]])


    See Also
    --------
    roll_max

    """
    return _roll_min(X, w)


def _roll_min(X, w):
    if len(X.shape) == 2:

        return np.asarray(roll_min_cy_2d(X, w))

    return np.asarray(roll_min_cy_1d(X, w))


@WrapperArray('dtype', 'axis', 'window')
def roll_max(X, w=None, axis=0, dtype=None):
    r""" Compute simple rolling maximum of size `w` for each `X`' series.

    .. math::

        roll_max^w_t(X) = max(X_{t - w}, ..., X_t)

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the rolling maximum.
    w : int, optional
        Size of the lagged window of the rolling maximum, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is None.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Simple rolling maximum of each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> roll_max(X, w=3, dtype=np.float64, axis=0)
    array([ 60., 100., 100., 120., 160., 160.])
    >>> X = np.array([[60, 60], [100, 100], [80, 80],
    ...               [120, 120], [160, 160], [80, 80]])
    >>> roll_max(X, w=3, dtype=np.float64, axis=0)
    array([[ 60.,  60.],
           [100., 100.],
           [100., 100.],
           [120., 120.],
           [160., 160.],
           [160., 160.]])
    >>> roll_max(X, w=3, dtype=np.float64, axis=1)
    array([[ 60.,  60.],
           [100., 100.],
           [ 80.,  80.],
           [120., 120.],
           [160., 160.],
           [ 80.,  80.]])


    See Also
    --------
    roll_max

    """
    return _roll_max(X, w)


def _roll_max(X, w):
    if len(X.shape) == 2:

        return np.asarray(roll_max_cy_2d(X, w))

    return np.asarray(roll_max_cy_1d(X, w))
