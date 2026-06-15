#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-18 21:15:59
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-18 22:21:22

""" Rolling extremum functions.

Rolling minimum and maximum over a lagged window. Used directly as
features (e.g. price-channel breakouts) and internally by
:func:`fynance.features.scale.roll_normalize` to compute lookahead-safe
min-max scaling parameters.

Main entry points
-----------------
- :func:`roll_min` — rolling minimum over a window.
- :func:`roll_max` — rolling maximum over a window.

"""

from __future__ import annotations

# Built-in packages
# Third party packages
import numpy as np

# Local packages
from numba import njit
from numpy.typing import NDArray

from fynance._wrappers import WrapperArray

__all__ = ["roll_min", "roll_max"]


@njit(cache=True)
def _roll_min_1d(X, w):
    """ Rolling minimum over a trailing window of size ``w`` (numba). """
    T = X.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        i = max(0, t - w + 1)
        m = X[i]
        while i < t:
            i += 1
            if X[i] < m:
                m = X[i]
        out[t] = m
    return out


@njit(cache=True)
def _roll_min_2d(X, w):
    """ Column-wise rolling minimum (numba). """
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        for t in range(T):
            i = max(0, t - w + 1)
            m = X[i, n]
            while i < t:
                i += 1
                if X[i, n] < m:
                    m = X[i, n]
            out[t, n] = m
    return out


@njit(cache=True)
def _roll_max_1d(X, w):
    """ Rolling maximum over a trailing window of size ``w`` (numba). """
    T = X.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        i = max(0, t - w + 1)
        m = X[i]
        while i < t:
            i += 1
            if X[i] > m:
                m = X[i]
        out[t] = m
    return out


@njit(cache=True)
def _roll_max_2d(X, w):
    """ Column-wise rolling maximum (numba). """
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        for t in range(T):
            i = max(0, t - w + 1)
            m = X[i, n]
            while i < t:
                i += 1
                if X[i, n] > m:
                    m = X[i, n]
            out[t, n] = m
    return out



# =========================================================================== #
#                                   Min Max                                   #
# =========================================================================== #


@WrapperArray('dtype', 'axis', 'window')
def roll_min(X: NDArray, w: int | None = None, axis: int = 0, dtype=None) -> NDArray:
    r""" Compute simple rolling minimum of size `w` for each `X`' series.

    .. math::

        roll\_min^w_t(X) = min(X_{t - w}, ..., X_t)

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

        return _roll_min_2d(X, w)

    return _roll_min_1d(X, w)


@WrapperArray('dtype', 'axis', 'window')
def roll_max(X: NDArray, w: int | None = None, axis: int = 0, dtype=None) -> NDArray:
    r""" Compute simple rolling maximum of size `w` for each `X`' series.

    .. math::

        roll\_max^w_t(X) = max(X_{t - w}, ..., X_t)

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

        return _roll_max_2d(X, w)

    return _roll_max_1d(X, w)
