#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-07-24 15:11:52
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-16 18:42:06
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


cpdef double [:] smstd_cy_1d(double [:] X, int w):
    """ Compute simple moving standard deviation of one-dimensional array.

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
        Simple moving standard deviation. Can be converted to a NumPy array, C
        array, Cython array, etc.

    """
    cdef int T, t = 0

    T = X.shape[0]
    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] sd = var
    cdef double m, m2, S2, S
    S = <double>0
    S2 = <double>0

    while t < T:
        if t < w:
            S += X[t]
            S2 += X[t] ** <double>2
            m = S / <double>(t + 1)
            m2 = S2 / <double>(t + 1)
            sd[t] = sqrt(m2 - m ** <double>2)

        else:
            S += X[t] - X[t - w]
            S2 += X[t] ** <double>2 - X[t - w] ** <double>2
            m = S / <double>w
            m2 = S2 / <double>w
            sd[t] = sqrt(m2 - m ** <double>2)

        t += 1
    
    return sd


cpdef double [:, :] smstd_cy_2d(double [:, :] X, int w):
    """ Compute simple moving standard deviations of two-dimensional array
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
        Simple moving standard deviation. Can be converted to a NumPy array, C
        array, Cython array, etc.

    """
    cdef int T, t, N, n = 0

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] sd = var
    cdef double m, m2, S2, S

    while n < N:
        t = 0
        S = <double>0
        S2 = <double>0
        while t < T:
            if t < w:
                S += X[t, n]
                S2 += X[t, n] ** <double>2
                m = S / <double>(t + 1)
                m2 = S2 / <double>(t + 1)
                sd[t, n] = sqrt(m2 - m ** <double>2)

            else:
                S += X[t, n] - X[t - w, n]
                S2 += X[t, n] ** <double>2 - X[t - w, n] ** <double>2
                m = S / <double>w
                m2 = S2 / <double>w
                sd[t, n] = sqrt(m2 - m ** <double>2)

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


# =========================================================================== #
#                                 Old Script                                  #
# =========================================================================== #


cpdef np.ndarray[np.float64_t, ndim=1] sma_cy(
        np.ndarray[np.float64_t, ndim=1] X, 
        int k=21,
    ):
    """ Simple moving average of k lags. Vectorized method. """
    cdef np.ndarray[np.float64_t, ndim=1] ma

    ma = np.cumsum(X, dtype=np.float64)
    ma[k: ] = (ma[k: ] - ma[: -k]) / <np.float64_t>k
    ma[: k] /= np.arange(1, k + 1, dtype=np.float64)

    return ma


cpdef np.ndarray[np.float64_t, ndim=2] sma_cynp_2d(
        np.ndarray[np.float64_t, ndim=2] X, 
        int k=21,
    ):
    """ Simple moving average of k lags. Vectorized method. """
    cdef np.ndarray[np.float64_t, ndim=2] ma, arange

    arange = np.arange(1, k + 1, dtype=np.float64).reshape([k, 1])

    ma = np.cumsum(X, dtype=np.float64, axis=0)
    ma[k: ] = (ma[k: ] - ma[: -k]) / <np.float64_t>k
    ma[: k] /= arange # np.arange(1, k + 1, dtype=np.float64)

    return ma


cpdef np.ndarray[np.float64_t, ndim=1] wma_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        int lags=21,
    ):
    """ Weighted moving average of k lags. """
    cdef int t, t_0, t_T, T = series.size
    cdef double m
    cdef np.ndarray[np.float64_t, ndim=1] ma, arange

    arange = np.arange(1, T, dtype=np.float64)
    ma = np.zeros([T], dtype=np.float64)

    for t in range(T):
        m = <double>min(t + 1, lags)
        m = m * (m + 1.) / 2.
        t_0 = max(t - lags + 1, 0)
        t_T =  min(t + 1, lags)

        ma[t] = sum(arange[: t_T] * series[t_0: t + 1])
        ma[t] = ma[t] / m

    return ma


cpdef np.ndarray[np.float64_t, ndim=1] ema_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        float alpha=0.94,
    ):
    """ Exponential moving average. """
    cdef int t, T=series.size
    cdef np.ndarray[np.float64_t, ndim=1] ema=np.zeros([T], dtype=np.float64)
    
    ema[0] = series[0]

    for t in range(1, T):
        ema[t] = alpha * ema[t - 1] + (1. - alpha) * series[t]

    return ema


cpdef np.ndarray[np.float64_t, ndim=1] smstd_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        int lags=21,
    ):
    """ Simple moving standard deviation along k lags. """
    cdef int t, T = series.size
    cdef np.ndarray[np.float64_t, ndim=1] ma = sma_cy(series, k=lags)
    cdef np.ndarray[np.float64_t, ndim=1] std = np.zeros([T], dtype=np.float64)
    
    for t in range(T):
        std[t] = np.sum(
            np.square(series[max(t-lags+1, 0): t+1] - ma[t], dtype=np.float64),
            dtype=np.float64
            )  / <double>min(t + 1, lags)

    return np.sqrt(std, dtype=np.float64)


cpdef np.ndarray[np.float64_t, ndim=1] wmstd_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        int lags=21
    ):
    """ Weighted moving standard deviation along k lags. """
    cdef int t, T = series.size
    cdef float m
    cdef np.ndarray[np.float64_t, ndim=1] ma = wma_cy(series, lags=lags)
    cdef np.ndarray[np.float64_t, ndim=1] std = np.zeros([T], dtype=np.float)
    
    for t in range(T):
        m = <double>min(t + 1, lags)
        std[t] = np.sum(
            np.arange(1., m + 1., dtype=np.float64) \
            * (series[max(t - lags + 1, 0): t + 1] - ma[t]) ** 2 \
            / (m * (m + 1.) / 2.), dtype=np.float64
        )

    return np.sqrt(std, dtype=np.float64)


cpdef np.ndarray[np.float64_t, ndim=1] emstd_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        float alpha=0.94,
    ):
    """ Exponential moving standard deviation. """
    cdef t, T = series.size
    cdef np.ndarray[np.float64_t, ndim=1] ma = ema_cy(series, alpha=alpha)
    cdef np.ndarray[np.float64_t, ndim=1] std = np.zeros([T], dtype=np.float64)

    for t in range(1, T):
        std[t] = alpha * std[t-1] + (1. - alpha) * (series[t] - ma[t]) ** 2

    return np.sqrt(std, dtype=np.float64)
