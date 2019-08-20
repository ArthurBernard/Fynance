#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-20 19:57:33
# @Last modified by: ArthurBernard
# @Last modified time: 2019-08-20 14:46:31

""" Indicators functions. """

# Built-in packages

# External packages
import numpy as np

# Local packages
from fynance.tools.momentums import sma, ema, wma, smstd, emstd, wmstd
from fynance.tools.metrics import roll_mad

__all__ = [
    'bollinger_band', 'cci', 'hma', 'macd_hist', 'macd_line',
    'rsi', 'signal_line',
]


# =========================================================================== #
#                                 Indicators                                  #
# =========================================================================== #


def bollinger_band(series, lags=21, n_std=2, kind_ma='sma'):
    """ Compute the bollinger bands.

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Series.
    lags : int, optional
        Number of lags for ma, default is 21.
    n_std : float (default 1)
        Number of standard deviation, default is 1.
    kind_ma : str {'ema', 'sma', 'wma'}, optional
        Kind of moving average, default is 'ema'.

    Returns
    -------
    ma : np.ndarray[dtype=np.float64, ndim=1]
        Moving average of series.
    std : np.ndarray[dtype=np.float64, ndim=1]
        `n_std` moving standard deviation of series.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> mean_vect, std_vect = bollinger_band(series, lags=3)
    >>> mean_vect
    array([ 60.,  80.,  80., 100., 120., 120.])
    >>> std_vect
    array([ 0.        , 40.        , 32.65986324, 32.65986324, 65.31972647,
           65.31972647])

    See Also
    --------
    z_score, rsi, hma, macd_hist, cci

    """
    if kind_ma.lower() == 'sma':
        ma = sma(series, lags=lags)
        std = smstd(series, lags=lags)

    elif kind_ma.lower() == 'ema':
        ma = ema(series, lags=lags)
        std = emstd(series, lags=lags)

    elif kind_ma.lower() == 'wma':
        ma = wma(series, lags=lags)
        std = wmstd(series, lags=lags)

    else:
        raise ValueError('Unknown kind_ma: {}'.format(kind_ma))

    return ma, n_std * std


def cci(series, high=None, low=None, lags=20):
    r""" Compute Commodity Channel Index _[1].

    Notes
    -----
    CCI is an oscillator introduced by Donald Lamber in 1980. It is calculated
    as the difference between the typical price of a commodity and its simple
    moving average, divided by the moving mean absolute deviation of the
    typical price. The index is usually scaled by an inverse factor of 0.015 to
    provide more readable numbers:

    .. math::

        cci = \frac{1}{0.015} \frac{p_t - sma(p_t)}{mad(p_t)}
        \text{where }p_t = \frac{p_{close} + p_{high} + p_{low}}{3}

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Series of close prices.
    high, low : np.ndarray[dtype=np.float64, ndim=1], optional
        Series of high and low prices, if `None` then `p_t` is computed with
        only closed prices.
    lags : int, optional
        Number of lags to compute the simple moving average and the mean
        absolute deviation.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Series of commodity channal index.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Commodity_channel_index

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> cci(series, lags=3)
    array([   0.        ,   66.66666667,    0.        ,  100.        ,
            100.        , -100.        ])

    See Also
    --------
    bollinger_band, rsi, hma, macd_hist

    """
    if high is None:
        high = series

    if low is None:
        low = series

    # Compute typical price
    p = (series + high + low) / 3
    # Compute moving mean absolute deviation
    r_mad = roll_mad(p, win=lags)
    # Avoid zero division
    r_mad[r_mad == 0.] = 1.

    return (p - sma(p, lags=lags)) / r_mad / 0.015


def hma(series, lags=21, kind_ma='wma'):
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
    kind_ma : str {'ema', 'sma', 'wma'}, optional
        Kind of moving average, default is 'ema'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Hull moving average of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> hma(series, lags=3)
    array([ 60.        , 113.33333333,  76.66666667, 136.66666667,
           186.66666667,  46.66666667])

    See Also
    --------
    z_score, bollinger_band, rsi, macd_hist, cci

    """
    if kind_ma.lower() == 'ema':
        f = ema

    elif kind_ma.lower() == 'sma':
        f = sma

    elif kind_ma.lower() == 'wma':
        f = wma

    else:
        raise ValueError('Unknown kind_ma: {}'.format(kind_ma))

    wma1 = f(series, lags=int(lags / 2))
    wma2 = f(series, lags=lags)
    hma = f(2 * wma1 - wma2, lags=int(np.sqrt(lags)))

    return hma


def macd_hist(series, lags=9, fast_ma=12, slow_ma=26, kind_ma='ema'):
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
    kind_ma : str {'ema', 'sma', 'wma'}, optional
        Kind of moving average, default is 'ema'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving avg convergence/divergence histogram of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> macd_hist(series, lags=3, fast_ma=2, slow_ma=4)
    array([ 0.        ,  5.33333361, -0.35555539,  3.9348151 ,  6.41027212,
           -9.47070965])

    See Also
    --------
    z_score, bollinger_band, hma, macd_line, signal_line, cci

    """
    macd_lin = macd_line(
        series, fast_ma=fast_ma, slow_ma=slow_ma, kind_ma=kind_ma
    )
    sig_lin = signal_line(
        series, lags=lags, fast_ma=fast_ma, slow_ma=slow_ma, kind_ma=kind_ma
    )
    hist = macd_lin - sig_lin

    return hist


def macd_line(series, fast_ma=12, slow_ma=26, kind_ma='ema'):
    """ Compute Moving Average Convergence Divergence Line.

    Parameters
    ----------
    series : np.ndarray[dtype=np.float64, ndim=1]
        Series of index or returns.
    fast_ma : int, optional
        Number of lags for short ma, default is 12.
    slow_ma : int, optional
        Number of lags for long ma, default is 26.
    kind_ma : str {'ema', 'sma', 'wma'}, optional
        Kind of moving average, default is 'ema'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Moving avg convergence/divergence line of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> macd_line(series, fast_ma=2, slow_ma=4)
    array([ 0.        , 10.66666722,  4.62222282, 12.84740842, 21.73313755,
           -3.61855386])

    See Also
    --------
    z_score, bollinger_band, hma, macd_hist, signal_line, cci

    """
    if kind_ma.lower() == 'wma':
        f = wma

    elif kind_ma.lower() == 'sma':
        f = sma

    elif kind_ma.lower() == 'ema':
        f = ema

    else:
        raise ValueError('Unknown kind_ma: {}'.format(kind_ma))

    fast = f(series, lags=fast_ma)
    slow = f(series, lags=slow_ma)
    macd_lin = fast - slow

    return macd_lin


def rsi(series, kind_ma='ema', lags=21, alpha=None):
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
    kind_ma : str {'ema', 'sma', 'wma'}, optional
        Kind of moving average, default is 'ema'.
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
    >>> series = np.array([60, 100, 80, 120, 160, 80])
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

    if kind_ma.lower() == 'sma':
        ma_U = sma(U, lags=lags)
        ma_D = sma(D, lags=lags)

    elif kind_ma.lower() == 'ema':
        ma_U = ema(U, lags=lags, alpha=alpha)
        ma_D = ema(D, lags=lags, alpha=alpha)

    elif kind_ma.lower() == 'wma':
        ma_U = wma(U, lags=lags)
        ma_D = wma(D, lags=lags)

    else:
        raise ValueError('Unknown kind_ma: {}'.format(kind_ma))

    RSI = np.zeros([T])
    RSI[1:] = 100 * ma_U / (ma_U + ma_D + 1e-8)

    return RSI


def signal_line(series, lags=9, fast_ma=12, slow_ma=26, kind_ma='ema'):
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
    kind_ma : str {'ema', 'sma', 'wma'}, optional
        Kind of moving average, default is 'ema'.

    Returns
    -------
    np.ndarray[dtype=np.float64, ndim=1]
        Signal line of index or returns.

    Examples
    --------
    >>> series = np.array([60, 100, 80, 120, 160, 80])
    >>> signal_line(series, lags=3, fast_ma=2, slow_ma=4)
    array([ 0.        ,  5.33333361,  4.97777822,  8.91259332, 15.32286544,
            5.85215579])

    See Also
    --------
    z_score, bollinger_band, hma, macd_hist, macd_line, cci

    """
    macd_lin = macd_line(series, fast_ma=fast_ma, slow_ma=slow_ma)

    if kind_ma.lower() == 'wma':
        f = wma

    elif kind_ma.lower() == 'sma':
        f = sma

    elif kind_ma.lower() == 'ema':
        f = ema

    else:
        raise ValueError('Unknown kind_ma: {}'.format(kind_ma))

    sig_lin = f(macd_lin, lags=lags)

    return sig_lin


if __name__ == '__main__':

    import doctest

    doctest.testmod()
