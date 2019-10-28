#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-07-09 10:49:19
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-28 11:27:32
# cython: language_level=3, wraparound=False, boundscheck=False

# Built-in packages
from libc.math cimport sqrt

# External packages
from cython cimport view
import numpy as np
cimport numpy as np

# Local packages
from fynance.tools.momentums_cy import smstd_cy, sma_cy, sma_cy_1d, sma_cy_2d

# TODO list:
# - Append window size on rolling MDD


__all__ = [
    'drawdown_cy_1d', 'drawdown_cy_2d', 
    'roll_drawdown_cy_1d', 'roll_drawdown_cy_2d',
    'roll_mad_cy_1d', 'roll_mad_cy_2d',
    'roll_mdd_cy_1d', 'roll_mdd_cy_2d',
]


# =========================================================================== #
#                                   Metrics                                   #
# =========================================================================== #


cpdef double [:] drawdown_cy_1d(double [:] X, int raw):
    """ Compute drawdown of a one-dimensional array.
    
    Measure of the decline from a historical peak in some variable [1]_
    (typically the cumulative profit or total open equity of a financial
    trading strategy). 

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    raw : {0, 1}
        If 1 compute the raw drawdown, otherwise compute drawdown in
        percentage.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Series of DrawDown. Can be converted to a NumPy array, C array,
        Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)

    """
    cdef int T = X.shape[0]

    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] dd = var
    cdef double one = <double>1
    cdef double S = X[0]
    cdef int t = 0

    # Compute DrawDown
    while t < T:
        S = max(S, X[t])
        if raw != 0:
            dd[t] = S - X[t]

        else:
            dd[t] = one - X[t] / S

        t += 1
    
    return dd


cpdef double [:, :] drawdown_cy_2d(double [:, :] X, int raw):
    """ Compute drawdown of a two-dimensional array.

    Measure of the decline from a historical peak in some variable [1]_
    (typically the cumulative profit or total open equity of a financial
    trading strategy). 

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    raw : {0, 1}
        If 1 compute the raw drawdown, otherwise compute drawdown in
        percentage.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Series of DrawDown. Can be converted to a NumPy array, C array,
        Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)

    """
    cdef int T = X.shape[0]
    cdef int N = X.shape[1]

    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] dd = var
    cdef double one = <double>1
    cdef double S
    cdef int t, n = 0

    # Compute DrawDown
    while n < N:
        S = X[0, n]
        t = 0
        while t < T:
            S = max(S, X[t, n])
            if raw != 0:
                dd[t, n] = S - X[t, n]

            else:
                dd[t, n] = one - X[t, n] / S

            t += 1

        n += 1
    
    return dd


# =========================================================================== #
#                               Rolling metrics                               #
# =========================================================================== #


cpdef double [:] roll_drawdown_cy_1d(double [:] X, int w, int raw):
    """ Compute the rolling Drawdown for one-dimensional array.

    Measure of the decline from a historical peak in some variable [1]_
    (typically the cumulative profit or total open equity of a financial
    trading strategy).

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the lagged window.
    raw : {0, 1}
        If 1 compute the raw drawdown, otherwise compute drawdown in
        percentage.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Series of DrawDown. Can be converted to a NumPy array, C array,
        Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)

    """
    cdef int i, T = X.shape[0]

    if w >= T:

        return drawdown_cy_1d(X, raw)

    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] dd = var
    cdef double one = <double>1
    cdef double S = X[0]
    cdef int t = 0

    # Compute DrawDown
    while t < T:
        if t < w:
            S = max(S, X[t])

        else:
            S = X[t]
            i = 1
            while i < w:
                S = max(S, X[t - i])
                i += 1

        if raw != 0:
            dd[t] = S - X[t]

        else:
            dd[t] = one - X[t] / S

        t += 1
    
    return dd


cpdef double [:, :] roll_drawdown_cy_2d(double [:, :] X, int w, int raw):
    """ Compute the rolling Drawdown for two-dimensional array.

    Measure of the decline from a historical peak in some variable [1]_
    (typically the cumulative profit or total open equity of a financial
    trading strategy).

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the lagged window.
    raw : {0, 1}
        If 1 compute the raw drawdown, otherwise compute drawdown in
        percentage.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Series of DrawDown. Can be converted to a NumPy array, C array,
        Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)

    """
    cdef int i, t, T = X.shape[0]

    if w >= T:

        return drawdown_cy_2d(X, raw)

    cdef int n, N = X.shape[1]

    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] dd = var
    cdef double S, one = <double>1
    n = 0

    # Compute DrawDown
    while n < N:
        t = 0
        S = X[0, n]
        while t < T:
            if t < w:
                S = max(S, X[t, n])

            else:
                S = X[t, n]
                i = 1
                while i < w:
                    S = max(S, X[t - i, n])
                    i += 1

            if raw != 0:
                dd[t, n] = S - X[t, n]

            else:
                dd[t, n] = one - X[t, n] / S

            t += 1

        n += 1

    return dd


cpdef double [:] roll_mad_cy_1d(double [:] X, int w):
    """ Compute rolling Mean Absolut Deviation for one-dimensional array.

    Compute the moving average of the absolute value of the distance to the
    moving average _[1].

    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Time series (price, performance or index). Can be a NumPy array, C
        array, Cython array, etc.
    w : int
        Window size.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Series of mean absolute deviation. Can be converted to a NumPy array,
        C array, Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Average_absolute_deviation

    """
    cdef int i, t, T = X.shape[0]

    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] ma = sma_cy_1d(X, w)
    cdef double [:] mad = var
    cdef double S

    t = 0
    while t < T:
        i = 0
        S = <double>0.
        if t < w:
            while i <= t:
                S += abs(X[i] - ma[t])
                i += 1

            mad[t] = S / <double>(t + 1)

        else:
            while i < w:
                S += abs(X[t - i] - ma[t])
                i += 1

            mad[t] = S / <double>w

        t += 1

    return mad


cpdef double [:, :] roll_mad_cy_2d(double [:, :] X, int w):
    """ Compute rolling Mean Absolut Deviation for two-dimensional array.

    Compute the moving average of the absolute value of the distance to the
    moving average _[1].

    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Time series (price, performance or index). Can be a NumPy array, C
        array, Cython array, etc.
    w : int
        Window size.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Series of mean absolute deviation. Can be converted to a NumPy array,
        C array, Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Average_absolute_deviation

    """
    cdef int i, T, t, N, n = 0 

    T = X.shape[0]
    N = X.shape[1]
    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] ma = sma_cy_2d(X, w)
    cdef double [:, :] mad = var
    cdef double S

    while n < N:
        t = 0
        while t < T:
            i = 0
            S = <double>0.
            if t < w:
                while i <= t:
                    S += abs(X[i, n] - ma[t, n])
                    i += 1

                mad[t, n] = S / <double>(t + 1)

            else:
                while i < w:
                    S += abs(X[t - i, n] - ma[t, n])
                    i += 1

                mad[t, n] = S / <double>w

            t += 1

        n += 1

    return mad


cpdef double [:] roll_mdd_cy_1d(double [:] X, int w, int raw):
    """ Compute rolling maximum drawdown for one-dimensional array.

    Function to compute the maximum drwdown where drawdown is the measure of 
    the decline from a historical peak in some variable [1]_ (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    X : memoryview.ndarray[ndim=1, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the lagged window.
    raw : {0, 1}
        If 1 compute the raw drawdown, otherwise compute drawdown in
        percentage.

    Returns
    -------
    memoryview.ndarray[ndim=1, dtype=double]
        Series of MaxDrawDown. Can be converted to a NumPy array, C array,
        Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)
    
    """
    cdef int i, T = X.shape[0]

    var = view.array(shape=(T,), itemsize=sizeof(double), format='d')

    cdef double [:] dd = roll_drawdown_cy_1d(X, w, raw)
    cdef double [:] mdd = var
    cdef double S
    cdef int t = 0

    while t < T:
        i = 1
        S = dd[t]
        if t < w:
            while i < t:
                S = max(S, dd[t - i])
                i += 1

        else:
            while i < w:
                S = max(S, dd[t - i])
                i += 1

        mdd[t] = S
        t += 1
    
    return mdd


cpdef double [:, :] roll_mdd_cy_2d(double [:, :] X, int w, int raw):
    """ Compute rolling maximum drawdown for two-dimensional array.

    Function to compute the maximum drwdown where drawdown is the measure of 
    the decline from a historical peak in some variable [1]_ (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    X : memoryview.ndarray[ndim=2, dtype=double]
        Elements to compute the function. Can be a NumPy array, C array, Cython
        array, etc.
    w : int
        Size of the lagged window.
    raw : {0, 1}
        If 1 compute the raw drawdown, otherwise compute drawdown in
        percentage.

    Returns
    -------
    memoryview.ndarray[ndim=2, dtype=double]
        Series of MaxDrawDown. Can be converted to a NumPy array, C array,
        Cython array, etc.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)
    
    """
    cdef int i, T = X.shape[0]
    cdef int t, N = X.shape[1]

    var = view.array(shape=(T, N), itemsize=sizeof(double), format='d')

    cdef double [:, :] dd = roll_drawdown_cy_2d(X, w, raw)
    cdef double [:, :] mdd = var
    cdef double S
    cdef int n = 0

    while n < N:
        t = 0
        while t < T:
            i = 1
            S = dd[t, n]
            if t < w:
                while i < t:
                    S = max(S, dd[t - i, n])
                    i += 1

            else:
                while i < w:
                    S = max(S, dd[t - i, n])
                    i += 1

            mdd[t, n] = S
            t += 1

        n += 1

    return mdd


# =========================================================================== #
#                                 Old Scripts                                 #
# =========================================================================== #


cpdef np.float64_t sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        np.float64_t period=252.
    ):
    """ Compute Sharpe ratio.

    Function to compute the total return over the volatility, known as the 
    Sharpe ratio.
    
    Parameters
    ----------
    series: numpy.ndarray(dim=1, dtype=float)
        Prices of the index.
    period: int (default: 252)
        Number of period per year.

    Returns
    -------
    np.float64
        Value of Sharpe ratio.

    """
    if series[0] == 0.:
        return 0. 
    
    cdef np.float64_t T = <double>len(series)
    
    if T == 0.:
        return 0.
    
    # Compute compouned annual returns
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = series[1:] / series[:T-1] - 1.
    cdef np.float64_t annual_return 
    # TODO : `np.sign(series[-1] / series[0])` why i did this ? To fix
    annual_return = np.sign(series[T-1] / series[0]) * np.float_power(
        np.abs(series[T-1] / series[0]), <double>period / T, dtype=np.float64) - 1.
    
    # Compute annual volatility
    cdef np.float64_t vol = sqrt(<double>period) * np.std(ret_vect, dtype=np.float64)
    
    if vol == 0.:
        return 0.

    return annual_return / vol


cpdef np.float64_t log_sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        np.float64_t period=252.
    ):
    """ Compute Sharpe ratio.

    Function to compute the total return over the volatility, known as the 
    Sharpe ratio.
    
    Parameters
    ----------
    series: numpy.ndarray(dim=1, dtype=float)
        Prices of the index.
    period: int (default: 252)
        Number of period per year.

    Returns
    -------
    np.float64
        Value of Sharpe ratio.

    """
    
    cdef np.float64_t T = series.size 

    # Compute compouned annual returns
    cdef np.ndarray[np.float64_t, ndim=1] ret_vect = np.log(series[1:] / series[:T-1], dtype=np.float64)
    cdef np.float64_t annual_return = (<double>period / T) * np.sum(ret_vect, dtype=np.float64)
    
    # Compute annual volatility
    cdef np.float64_t vol = sqrt(<double>period) * np.std(ret_vect, dtype=np.float64)

    if vol == 0.:
        return 0.

    return annual_return / vol


cpdef np.ndarray[np.float64_t, ndim=1] roll_mad_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        int win=0
    ):
    """ Compute rolling Mean Absolut Deviation.

    Compute the moving average of the absolute value of the distance to the
    moving average _[1].

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Series of mean absolute deviation.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Average_absolute_deviation

    """
    cdef int t, T = series.size
    cdef np.ndarray[np.float64_t, ndim=1] ma = np.asarray(sma_cy_1d(series, win), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] mad = np.zeros([T], dtype=np.float64)
    
    for t in range(T):
        mad[t] = np.sum(
            np.abs(series[max(t-win+1, 0): t+1] - ma[t], dtype=np.float64),
            dtype=np.float64
            )  / <double>min(t + 1, win)

    return mad


cpdef np.ndarray[np.float64_t, ndim=1] drawdown_cy(
        np.ndarray[np.float64_t, ndim=1] series
    ):
    """ Compute drawdown of a series.
    
    Measure of the decline from a historical peak in some variable [1]_
    (typically the cumulative profit or total open equity of a financial
    trading strategy). 

    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Series of DrawDown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)

    """
    cdef np.ndarray[np.float64_t, ndim=1] maximums
    
    # Compute DrawDown
    maximums = np.maximum.accumulate(series, dtype=np.float64)
    
    return 1. - series / maximums


cpdef np.float64_t mdd_cy(np.ndarray[np.float64_t, ndim=1] series):
    """ Compute the maximum drawdown.

    Function to compute the maximum drwdown where drawdown is the measure of 
    the decline from a historical peak in some variable [1]_ (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).

    Returns
    -------
    np.float64
        Value of Maximum DrawDown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)
    
    """
    cdef np.ndarray[np.float64_t, ndim=1] drawdowns
    
    # Compute DrawDown
    drawdowns = drawdown_cy(series)

    return max(drawdowns)


cpdef np.float64_t calmar_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        np.float64_t period=252.
    ):
    """ Compute Calmar ratio.

    Function to compute the compouned annual return over the Maximum DrawDown, 
    known as the Calmar ratio.
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    period : int (default 252)
        Number of period per year.

    Returns
    -------
    np.float64
        Value of Calmar ratio.

    """
    cdef np.float64_t ret, annual_return, max_dd, T = series.size
    
    # Compute compouned annual returns
    ret = series[T-1] / series[0]
    annual_return = np.sign(ret) * np.float_power(
        np.abs(ret), period / T, dtype=np.float64) - 1.
    
    # Compute MaxDrawDown
    max_dd = mdd_cy(series)

    if max_dd == 0.:
        return 0.

    return annual_return / max_dd

cpdef np.ndarray[np.float64_t, ndim=1] roll_mdd_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        int win=0
    ):
    """ Compute rolling maximum drawdown.

    Function to compute the maximum drwdown where drawdown is the measure of 
    the decline from a historical peak in some variable [1]_ (typically the 
    cumulative profit or total open equity of a financial trading strategy). 
    
    Parameters
    ----------
    series : np.ndarray[np.float64, ndim=1]
        Time series (price, performance or index).
    win : int (default 0)
        Size of the rolling window. If less of two, 
        rolling Max DrawDown is compute on all the past.

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Series of rolling Maximum DrawDown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)
    
    """
    cdef np.ndarray[np.float64_t, ndim=1] drawdowns
    
    # Compute DrawDown
    drawdowns = drawdown_cy(series)
    
    if win <= 2:

        return np.maximum.accumulate(drawdowns, dtype=np.float64)


cpdef np.ndarray[np.float64_t, ndim=1] roll_sharpe_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        np.float64_t period=252.,
        int window=0
    ):
    """
    Rolling sharpe
    /! Not optimized /!
    """
    cdef int t, T = len(series)
    cdef np.ndarray[np.float64_t, ndim=1] roll_s = np.zeros([T], dtype=np.float64)
    
    for t in range(1, T):
        if window > t or window < 2:
            roll_s[t] = sharpe_cy(series[:t + 1], period=period)
        else:
            roll_s[t] = sharpe_cy(series[t - window: t + 1], period=period)

    return roll_s
