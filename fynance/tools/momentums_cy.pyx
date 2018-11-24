import numpy as np
cimport numpy as np


#=============================================================================#
#                               Moving Averages                               #
#=============================================================================#


cpdef np.ndarray[np.float64_t, ndim=1] sma_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        int lags=21,
    ):
    """
    Simple moving average of k lags. Vectorized method.
    """
    cdef np.ndarray[np.float64_t, ndim=1] ma

    ma = np.cumsum(series, dtype=np.float64)
    ma[lags: ] = (ma[lags: ] - ma[: -lags]) / <double>lags
    ma[: lags] /= np.arange(1, lags + 1, dtype=np.float64)
    return ma


cpdef np.ndarray[np.float64_t, ndim=1] wma_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        int lags=21,
    ):
    """ 
    Weighted moving average of k lags.
    """
    cdef int t, T = series.size
    cdef float m
    cdef np.ndarray[np.float64_t, ndim=1] ma = np.zeros([T], dtype=np.float64)

    for t in range(T):
        m = <float>min(t+1, lags)
        ma[t] = np.sum(
            np.arange(1., m + 1., dtype=np.float64) \
            * series[max(t-lags+1, 0): t+1] / (m * (m + 1.) / 2.), 
            dtype=np.float64
        )
    return ma


cpdef np.ndarray[np.float64_t, ndim=1] ema_cy(
        np.ndarray[np.float64_t, ndim=1] series,
        float alpha=0.94,
    ):
    """ 
    Exponential moving average.
    """
    cdef int t, T=series.size
    cdef np.ndarray[np.float64_t, ndim=1] ema=np.zeros([T], dtype=np.float64)
    
    ema[0] = series[0]
    for t in range(1, T):
        ema[t] = alpha * ema[t-1] + (1. - alpha) * series[t]
    return ema


#=============================================================================#
#                          Moving Standard Deviation                          #
#=============================================================================#


cpdef np.ndarray[np.float64_t, ndim=1] smstd_cy(
        np.ndarray[np.float64_t, ndim=1] series, 
        int lags=21,
    ):
    """ 
    Simple moving standard deviation along k lags. 
    """
    cdef int t, T = series.size
    cdef np.ndarray[np.float64_t, ndim=1] ma = sma_cy(series, lags=lags)
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
    """
    Weighted moving standard deviation along k lags.
    """
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
    """ 
    Exponential moving standard deviation. 
    """
    cdef t, T = series.size
    cdef np.ndarray[np.float64_t, ndim=1] ma = ema_cy(series, alpha=alpha)
    cdef np.ndarray[np.float64_t, ndim=1] std = np.zeros([T], dtype=np.float64)

    for t in range(1, T):
        std[t] = alpha * std[t-1] + (1. - alpha) * (series[t] - ma[t-1]) ** 2
    return np.sqrt(std, dtype=np.float64)