#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-20 19:57:33
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-16 13:09:06

""" Indicators functions. """

# Built-in packages
from warnings import warn

# External packages
import numpy as np

# Local packages
from fynance.tools._wrappers import WrapperArray
from fynance.tools.momentums import sma, ema, wma, smstd, emstd, wmstd, _sma
from fynance.tools.metrics import roll_mad, mad
from fynance.tools.metrics_cy import roll_mad_cy

__all__ = [
    'bollinger_band', 'cci', 'hma', 'macd_hist', 'macd_line',
    'rsi', 'signal_line',
]

_handler_ma = {'s': sma, 'w': wma, 'e': ema}
_handler_mstd = {'s': smstd, 'w': wmstd, 'e': emstd}

# =========================================================================== #
#                                 Indicators                                  #
# =========================================================================== #


@WrapperArray('dtype', 'axis', 'lags')
def bollinger_band(X, k=20, n=2, kind='s', axis=0, dtype=None):
    r""" Compute the bollinger bands for `k` lags for each `X`' series'.

    Let :math:`\mu_t` the moving average and :math:`\sigma_t` is the moving
    standard deviation of `X`.

    .. math::
        upperBand_t = \mu_t + n \times \sigma_t
        lowerBand_t = \mu_t - n \times \sigma_t


    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the bollinger bands. If `X` is a two-dimensional
        array, bollinger bands are computed for each series along `axis`.
    k : int, optional
        Number of lags used for computation, must be positive. Default is 20.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.
    n : float, optional
        Number of standard deviations above and below the moving average.
        Default is 2.
    kind : {'s', 'e', 'w'}
        Kind of moving average/standard deviation. Default is 's'.
        - Exponential moving average/standard deviation if 'e'.
        - Simple moving average/standard deviation if 's'.
        - Weighted moving average/standard deviation if 'w'.

    Returns
    -------
    upper_band, lower_band : np.ndarray[dtype, ndim=1 or 2]
        Respectively upper and lower bollinger bands for each series.

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> upper_band, lower_band = bollinger_band(X, k=3, n=2)
    >>> upper_band
    array([ 60.        , 120.        , 112.65986324, 132.65986324,
           185.31972647, 185.31972647])
    >>> lower_band
    array([60.        , 40.        , 47.34013676, 67.34013676, 54.68027353,
           54.68027353])

    See Also
    --------
    z_score, rsi, hma, macd_hist, cci

    """
    warn('Since version 1.1.0, bollinger_band returns upper and lower bands.')
    avg = _handler_ma[kind.lower()](X, k=k)
    std = _handler_mstd[kind.lower()](X, k=k)

    return avg + n * std, avg - n * std


@WrapperArray('dtype', 'axis')
def cci(X, high=None, low=None, k=20, axis=0, dtype=None):
    r""" Compute Commodity Channel Index for `k` lags for each `X`' series'.

    Notes
    -----
    CCI is an oscillator introduced by Donald Lamber in 1980 [1]_. It is
    calculated as the difference between the typical price of a commodity and
    its simple moving average, divided by the moving mean absolute deviation of
    the typical price. The index is usually scaled by an inverse factor of
    0.015 to provide more readable numbers:

    .. math::

        cci = \frac{1}{0.015} \frac{p_t - sma(p_t)}{mad(p_t)}
        \text{where }p_t = \frac{p_{close} + p_{high} + p_{low}}{3}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Series of close prices.
    high, low : np.ndarray[dtype, ndim=1 or 2], optional
        Series of high and low prices, if `None` then `p_t` is computed with
        only closed prices. Must have the same shape as `X`.
    k : int, optional
        Number of lags used to compute moving average, must be positive.
        Default is 20.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Commodity Channal Index for each series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Commodity_channel_index

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> cci(X, k=3, dtype=np.float64)
    array([   0.        ,   66.66666667,    0.        ,  100.        ,
            100.        , -100.        ])

    See Also
    --------
    bollinger_band, rsi, hma, macd_hist

    """
    if high is None:
        high = X

    if low is None:
        low = X

    # Compute typical price
    p = (X + high + low) / 3
    # Compute moving mean absolute deviation
    r_mad = roll_mad(p, win=k)
    # Avoid zero division
    r_mad[r_mad == 0.] = 1.

    return (p - _sma(p, k)) / r_mad / 0.015


def hma(series, lags=21, kind='w'):
    r""" Compute Hull Moving Average.

    Notes
    -----
    .. math:: hma = wma(2 \times wma(x, \frac{k}{2}) - wma(x, k), \sqrt{k})

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Series of prices or returns.
    lags : int, optional
        Number of lags for ma, default is 21.
    kind_ma : {'e', 's', 'w'}
        Kind of moving average, default is 'w'.
        - Exponential moving average if 'e'.
        - Simple moving average if 'sma'.
        - Weighted moving average if 'w'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Hull moving average of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> hma(series, lags=3)
    array([ 60.        , 113.33333333,  76.66666667, 136.66666667,
           186.66666667,  46.66666667])

    See Also
    --------
    z_score, bollinger_band, rsi, macd_hist, cci

    """
    f = _handler_ma[kind.lower()]

    wma1 = f(series, k=int(lags / 2))
    wma2 = f(series, k=lags)
    hma = f(2 * wma1 - wma2, k=int(np.sqrt(lags)))

    return hma


def macd_hist(series, lags=9, fast_ma=12, slow_ma=26, kind='e'):
    """ Compute Moving Average Convergence Divergence Histogram.

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Series of index or returns.
    lags : int, optional
        Number of lags, default is 9.
    fast_ma : int, optional
        Number of lags for short ma, default is 12.
    slow_ma : int, optional
        Number of lags for long ma, default is 26.
    kind_ma : {'e', 's', 'w'}
        Kind of moving average, default is 'e'.
        - Exponential moving average if 'e'.
        - Simple moving average if 's'.
        - Weighted moving average if 'w'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving avg convergence/divergence histogram of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> macd_hist(series, lags=3, fast_ma=2, slow_ma=4)
    array([ 0.        ,  5.33333333, -0.35555556,  3.93481481,  6.4102716 ,
           -9.47070947])

    See Also
    --------
    z_score, bollinger_band, hma, macd_line, signal_line, cci

    """
    macd_lin = macd_line(
        series, fast_ma=fast_ma, slow_ma=slow_ma, kind=kind
    )
    sig_lin = signal_line(
        series, lags=lags, fast_ma=fast_ma, slow_ma=slow_ma, kind=kind
    )
    hist = macd_lin - sig_lin

    return hist


def macd_line(series, fast_ma=12, slow_ma=26, kind='e'):
    """ Compute Moving Average Convergence Divergence Line.

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Series of index or returns.
    fast_ma : int, optional
        Number of lags for short ma, default is 12.
    slow_ma : int, optional
        Number of lags for long ma, default is 26.
    kind_ma : {'e', 's', 'w'}
        Kind of moving average, default is 'e'.
        - Exponential moving average if 'e'.
        - Simple moving average if 's'.
        - Weighted moving average if 'w'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving avg convergence/divergence line of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> macd_line(series, fast_ma=2, slow_ma=4)
    array([ 0.        , 10.66666667,  4.62222222, 12.84740741, 21.7331358 ,
           -3.61855473])

    See Also
    --------
    z_score, bollinger_band, hma, macd_hist, signal_line, cci

    """
    f = _handler_ma[kind.lower()]

    fast = f(series, k=fast_ma)
    slow = f(series, k=slow_ma)
    macd_lin = fast - slow

    return macd_lin


def rsi(series, kind='e', lags=21, alpha=None):
    r""" Compute Relative Strenght Index.

    Notes
    -----
    It is the average gain of upward periods (noted `U`) divided by the average
    loss of downward (noted `D`) periods during the specified time frame, such
    that :

    .. math:: RSI = 100 - \frac{100}{1 + \frac{U}{D}}

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Index series.
    kind_ma : {'e', 's', 'w'}
        Kind of moving average, default is 'e'.
        - Exponential moving average if 'e'.
        - Simple moving average if 's'.
        - Weighted moving average if 'w'.
    lags : int, optional
        Number of lagged period, default is 21.
    alpha : float, optional
        Coefficiant, default is 0.94 corresponding at 20 lags days (only for
        'ema').

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Value of RSI for each period.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> rsi(series, lags=3)
    array([ 0.        , 99.99999804, 69.59769254, 85.55610891, 91.72201613,
           30.00294321])

    See Also
    --------
    z_score, bollinger_band, hma, macd_hist, cci

    """
    series = series.flatten()
    T = np.size(series)
    U = np.zeros([T - 1])
    D = np.zeros([T - 1])
    delta = np.log(series[1:] / series[:-1])
    U[delta > 0] = delta[delta > 0]
    D[delta < 0] = - delta[delta < 0]

    f = _handler_ma[kind.lower()]

    ma_U = f(U, k=lags)
    ma_D = f(D, k=lags)

    RSI = np.zeros([T])
    RSI[1:] = 100 * ma_U / (ma_U + ma_D + 1e-8)

    return RSI


def signal_line(series, lags=9, fast_ma=12, slow_ma=26, kind='e'):
    """ Signal Line for k lags with slow and fast lenght.

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Index or returns ?
    lags : int, optional
        Number of lags, default is 9.
    fast_ma : int, optional
        Number of lags for short ma, default is 12.
    slow_ma : int, optional
        Number of lags for long ma, default is 26.
    kind_ma : {'e', 's', 'w'}
        Kind of moving average, default is 'e'.
        - Exponential moving average if 'e'.
        - Simple moving average if 's'.
        - Weighted moving average if 'w'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Signal line of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> signal_line(series, lags=3, fast_ma=2, slow_ma=4)
    array([ 0.        ,  5.33333333,  4.97777778,  8.91259259, 15.3228642 ,
            5.85215473])

    See Also
    --------
    z_score, bollinger_band, hma, macd_hist, macd_line, cci

    """
    macd_lin = macd_line(series, fast_ma=fast_ma, slow_ma=slow_ma)

    f = _handler_ma[kind.lower()]

    sig_lin = f(macd_lin, k=lags)

    return sig_lin


if __name__ == '__main__':

    import doctest

    doctest.testmod()
