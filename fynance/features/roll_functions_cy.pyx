#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-18 21:12:58
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-19 09:43:04
# cython: language_level=3, wraparound=False, boundscheck=False

# Built-in packages

# Third party packages
from cython cimport view
import numpy as np
cimport numpy as np

# Local packages

__all__ = [
    "roll_min_cy_1d", "roll_min_cy_2d", "roll_max_cy_1d", "roll_max_cy_2d"
]

# =========================================================================== #
#                                   Min Max                                   #
# =========================================================================== #


cpdef double [:] _roll_min_cy(double [:] X):
    var = view.array(shape=(X.shape[0],), itemsize=sizeof(double), format='d')

    cdef double [:] roll_min = var
    cdef double _min = <double>X[0]
    cdef int t = 0

    while t < X.shape[0]:
        _min = min(X[t], _min)
        roll_min[t] = _min
        t += 1

    return roll_min


cpdef double [:] roll_min_cy_1d(double [:] X, int w):
    """ Compute simple rolling minimum of one-dimensional array.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the window used for computation, must be strictly positive.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Simple rolling minimum. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    var = view.array(shape=(X.shape[0],), itemsize=sizeof(double), format='d')

    cdef double [:] roll_min = var
    cdef double _min = <double>0.
    cdef int i, t = 0

    if w == X.shape[0]:

        return _roll_min_cy(X)

    while t < X.shape[0]:
        i = max(0, t - w + 1)
        _min = X[i]

        while i < t:
            i += 1
            _min = min(X[i], _min)

        roll_min[t] = _min
        t += 1

    return roll_min


cpdef double [:, :] roll_min_cy_2d(double [:, :] X, int w):
    """ Compute simple rolling minimum of two-dimensional array along 0 axis.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the window used for computation, must be strictly positive.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Simple rolling minimum. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    cdef int T, N

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] roll_min = var
    cdef double _min
    cdef int i, t, n = 0

    if w == X.shape[0]:
        while n < N:
            roll_min[:, n] = _roll_min_cy(X[:, n])

        return roll_min

    while n < N:
        t = 0
        while t < T:
            i = max(0, t - w + 1)
            _min = X[i, n]

            while i < t:
                i += 1
                _min = min(X[i, n], _min)

            roll_min[t, n] = _min
            t += 1

        n += 1

    return roll_min


cpdef double [:] _roll_max_cy(double [:] X):
    var = view.array(shape=(X.shape[0],), itemsize=sizeof(double), format='d')

    cdef double [:] roll_max = var
    cdef double _max = <double>X[0]
    cdef int t = 0

    while t < X.shape[0]:
        _max = max(X[t], _max)
        roll_max[t] = _max
        t += 1

    return roll_max


cpdef double [:] roll_max_cy_1d(double [:] X, int w):
    """ Compute simple rolling maximum of one-dimensional array.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the window used for computation, must be strictly positive.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Simple rolling maximum. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    var = view.array(shape=(X.shape[0],), itemsize=sizeof(double), format='d')

    cdef double [:] roll_max = var
    cdef double _max = <double>0.
    cdef int i, t = 0

    if w == X.shape[0]:

        return _roll_max_cy(X)

    while t < X.shape[0]:
        i = max(0, t - w + 1)
        _max = X[i]

        while i < t:
            i += 1
            _max = max(X[i], _max)

        roll_max[t] = _max
        t += 1

    return roll_max


cpdef double [:, :] roll_max_cy_2d(double [:, :] X, int w):
    """ Compute simple rolling maximum of two-dimensional array along 0 axis.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the window used for computation, must be strictly positive.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Simple rolling maximum. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    cdef int T, N

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] roll_max = var
    cdef double _max
    cdef int i, t, n = 0

    if w == X.shape[0]:
        while n < N:
            roll_max[:, n] = _roll_max_cy(X[:, n])

        return roll_max

    while n < N:
        t = 0
        while t < T:
            i = max(0, t - w + 1)
            _max = X[i, n]

            while i < t:
                i += 1
                _max = max(X[i, n], _max)

            roll_max[t, n] = _max
            t += 1

        n += 1

    return roll_max
