#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-07-24 15:11:52
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-18 21:52:31
# cython: language_level=3, wraparound=False, boundscheck=False

# Built-in packages
from libc.math cimport sqrt

# External packages
from cython cimport view
import numpy as np
cimport numpy as np

# Local packages

__all__ = [
    'sma_cy_1d', 'sma_cy_2d', 'wma_cy_1d', 'wma_cy_2d',
    'ema_cy_1d', 'ema_cy_2d', 'smstd_cy_1d', 'smstd_cy_2d',
    'wmstd_cy_1d', 'wmstd_cy_2d', 'emstd_cy_1d', 'emstd_cy_2d',
]

# FIXME : problem with window size => window is w + 1 instead of w

# =========================================================================== #
#                               Moving Averages                               #
# =========================================================================== #


cpdef double [:] sma_cy_1d(double [:] X, int w):
    """ Compute simple moving average of one-dimensional array.

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
        Simple moving average. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    var = view.array(shape=(X.shape[0],), itemsize=sizeof(double), format='d')

    cdef double [:] ma = var
    cdef double S = <double>0.
    cdef int t = 0

    while t < X.shape[0]:
        if t < w:
            S += X[t]
            ma[t] = S / <double>(t + 1)

        else:
            S += X[t] - X[t - w]
            ma[t] = S / <double>w

        t += 1
    
    return ma


cpdef double [:, :] sma_cy_2d(double [:, :] X, int w):
    """ Compute simple moving averages of two-dimensional array along 0 axis.

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
        Simple moving average. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    cdef int T, N

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] ma = var
    cdef double S
    cdef int t, n = 0

    while n < N:
        t = 0
        S = <double>0.
        while t < T:
            if t < w:
                S += X[t, n]
                ma[t, n] = S / <double>(t + 1)

            else:
                S += X[t, n] - X[t - w, n]
                ma[t, n] = S / <double>w

            t += 1

        n += 1
    
    return ma


cpdef double [:] wma_cy_1d(double [:] X, int w):
    """ Compute weighted moving average of one-dimensional array.

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
        Weighted moving average. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    cdef int T, i, t = 0

    T = X.shape[0]
    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] ma = var
    cdef double S, m

    while t < T:
        i = 0
        S = <double>0.
        if t < w:
            m = <double>(t + 1)
            m = m * (m + <double>1) / <double>2
            while i <= t:
                S += <double>(i + 1) * X[i]
                i += 1

        else:
            while i < w:
                S += <double>(w - i) * X[t - i]
                i += 1

        ma[t] = S / m
        t += 1

    return ma


cpdef double [:, :] wma_cy_2d(double [:, :] X, int w):
    """ Compute weighted moving averages of two-dimensional array along 0 axis.

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
        Weighted moving average. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    cdef int T, N

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] ma = var
    cdef double S, m
    cdef int t, i, n = 0

    while n < N:
        t = 0
        while t < T:
            i = 0
            S = <double>0.
            if t < w:
                m = <double>(t + 1)
                m = m * (m + <double>1) / <double>2
                while i <= t:
                    S += <double>(i + 1) * X[i, n]
                    i += 1

            else:
                while i < w:
                    S += <double>(w - i) * X[t - i, n]
                    i += 1

            ma[t, n] = S / m
            t += 1

        n += 1

    return ma


cpdef double [:] ema_cy_1d(double [:] X, double alpha):
    """ Compute exponential moving average of one-dimensional array.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    alpha : double
        Multiplier, must be between 0 and 1.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Exponential moving average. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    cdef int t, T = X.shape[0]

    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] ma = var

    t = 1
    ma[0] = X[0]
    while t < T:
        ma[t] = alpha * ma[t - 1] + (1. - alpha) * X[t]
        t += 1

    return ma


cpdef double [:, :] ema_cy_2d(double [:, :] X, double alpha):
    """ Compute exponential moving averages of two-dimensional array along 0
    axis.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    alpha : double
        Multiplier, must be between 0 and 1.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Exponential moving average. Can be converted to a NumPy array, C array,
        Cython array, etc.

    """
    cdef int t, N, T, n = 0

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] ma = var

    while n < N:
        t = 1
        ma[0, n] = X[0, n]
        while t < T:
            ma[t, n] = alpha * ma[t - 1, n] + (1. - alpha) * X[t, n]
            t += 1

        n += 1

    return ma


# =========================================================================== #
#                          Moving Standard Deviation                          #
# =========================================================================== #


cpdef double [:] smstd_cy_1d(double [:] X, int w, int d):
    """ Compute simple moving standard deviation of one-dimensional array.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the window used for computation, must be strictly positive.
    d : int
        Number of degrees of freedom. Must be less than `w`.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Simple moving standard deviation. Can be converted to a NumPy array, C
        array, Cython array, etc.

    """
    cdef int t = 0, T = X.shape[0]

    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] sd = var
    cdef double S = <double>0, S2 = <double>0, sub_X
    cdef double _w = <double>1, _w_d = <double>1

    while t < T:
        if t < w:
            _w = <double>(t + 1)
            _w_d = <double>(t + 1 - d)
            sub_X = <double>0

        else:
            sub_X = X[t - w]

        S += X[t] - sub_X
        S2 += X[t] * X[t] - sub_X * sub_X
        
        if t < d:
            sd[t] = <double>0

        else:
            sd[t] = sqrt((S2 - (S / _w) * S) / _w_d)

        t += 1
    
    return sd


cpdef double [:, :] smstd_cy_2d(double [:, :] X, int w, int d):
    """ Compute simple moving standard deviations of two-dimensional array
    along 0 axis.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the window used for computation, must be strictly positive.
    d : int
        Number of degrees of freedom. Must be less than `w`.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Simple moving standard deviation. Can be converted to a NumPy array, C
        array, Cython array, etc.

    """
    cdef int t = 0, n = 0, T = X.shape[0], N = X.shape[1]

    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] sd = var
    cdef double S, S2, sub_X, _w = <double>1, _w_d = <double>1

    while n < N:
        t = 0
        S = <double>0
        S2 = <double>0

        while t < T:
            if t < w:
                _w = <double>(t + 1)
                _w_d = <double>(t + 1 - d)
                sub_X = <double>0

            else:
                sub_X = X[t - w, n]

            S += X[t, n] - sub_X
            S2 += X[t, n] * X[t, n] - sub_X * sub_X

            if t < d:
                sd[t, n] = <double>0

            else:
                sd[t, n] = sqrt((S2 - (S / _w) * S) / _w_d)

            t += 1

        n += 1
    
    return sd


cpdef double [:] wmstd_cy_1d(double [:] X, int w):
    """ Compute weighted moving standard deviation of one-dimensional array.

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
        Weighted moving standard deviation. Can be converted to a NumPy array,
        C array, Cython array, etc.

    """
    cdef int T, i, t = 0

    T = X.shape[0]
    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] sd = var
    cdef double S, S2, m

    while t < T:
        i = 0
        S = <double>0.
        S2 = <double>0.
        if t < w:
            m = <double>(t + 1)
            m = m * (m + <double>1) / <double>2
            while i <= t:
                S += <double>(i + 1) * X[i]
                S2 += <double>(i + 1) * X[i] ** <double>2
                i += 1

        else:
            while i < w:
                S += <double>(w - i) * X[t - i]
                S2 += <double>(w - i) * X[t - i] ** <double>2
                i += 1

        sd[t] = sqrt(S2 / m - (S / m) ** <double>2)
        t += 1

    return sd


cpdef double [:, :] wmstd_cy_2d(double [:, :] X, int w):
    """ Compute weighted moving standard deviation of two-dimensional array
    along 0 axis.

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
        Weighted moving standard deviation. Can be converted to a NumPy array,
        C array, Cython array, etc.

    """
    cdef int i, T, t, N, n = 0

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] sd = var
    cdef double S, S2, m

    while n < N:
        t = 0
        while t < T:
            i = 0
            S = <double>0.
            S2 = <double>0.
            if t < w:
                m = <double>(t + 1)
                m = m * (m + <double>1) / <double>2
                while i <= t:
                    S += <double>(i + 1) * X[i, n]
                    S2 += <double>(i + 1) * X[i, n] ** <double>2
                    i += 1

            else:
                while i < w:
                    S += <double>(w - i) * X[t - i, n]
                    S2 += <double>(w - i) * X[t - i, n] ** <double>2
                    i += 1

            sd[t, n] = sqrt(S2 / m - (S / m) ** <double>2)
            t += 1

        n += 1

    return sd


cpdef double [:] emstd_cy_1d(double [:] X, double alpha):
    """ Compute exponential moving standard deviation of one-dimensional array.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    alpha : double
        Multiplier, must be between 0 and 1.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Exponential moving standard deviation. Can be converted to a NumPy
        array, C array, Cython array, etc.

    """
    cdef int t, T = X.shape[0]

    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] sd = var
    cdef double m2, m = X[0]

    t = 1
    sd[0] = <double>0
    while t < T:
        m = alpha * m + (1. - alpha) * X[t]
        m2 = (1. - alpha) * (X[t] - m) ** <double>2
        sd[t] = sqrt(alpha * sd[t - 1] ** <double>2 + m2)
        t += 1

    return sd


cpdef double [:, :] emstd_cy_2d(double [:, :] X, double alpha):
    """ Compute exponential moving standard deviation of two-dimensional array
    along 0 axis.

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    alpha : double
        Multiplier, must be between 0 and 1.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Exponential moving standard deviation. Can be converted to a NumPy
        array, C array, Cython array, etc.

    """
    cdef int t, N, T, n = 0

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] sd = var
    cdef double m, m2

    while n < N:
        t = 1
        m = X[0, n]
        sd[0, n] = <double>0
        while t < T:
            m = alpha * m + (1. - alpha) * X[t, n]
            m2 = (1. - alpha) * (X[t, n] - m) ** <double>2
            sd[t, n] = sqrt(alpha * sd[t - 1, n] ** <double>2 + m2)
            t += 1

        n += 1

    return sd
