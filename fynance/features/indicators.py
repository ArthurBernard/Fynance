#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-20 19:57:33
# @Last modified by: ArthurBernard
# @Last modified time: 2019-11-18 17:57:42

""" Technical indicators for price-series analysis.

Classical technical-analysis indicators — Bollinger Band, CCI, Hull
moving average, MACD components, RSI — derived from rolling moving
averages and standard deviations (see :mod:`fynance.features.momentums`).

These indicators are commonly used to build trading signals or features
for machine-learning models in finance.

Main entry points
-----------------
- :func:`bollinger_band` — upper / lower volatility envelope.
- :func:`cci` — Commodity Channel Index.
- :func:`hma` — Hull Moving Average.
- :func:`macd_line`, :func:`signal_line`, :func:`macd_hist` — MACD.
- :func:`rsi` — Relative Strength Index.

"""

from __future__ import annotations

# Built-in packages
from warnings import warn

# External packages
import numpy as np
from numpy.typing import NDArray
from scipy import stats as _sp_stats

# Local packages
from fynance._wrappers import WrapperArray
from fynance.features.momentums import _ema, _emstd, _sma, _smstd, _wma, _wmstd
from fynance.features.stats import roll_mad

__all__ = [
    'bollinger_band', 'cci', 'hma', 'macd_hist', 'macd_line',
    'realized_volatility', 'roc', 'rolling_autocorr', 'rolling_kurtosis',
    'rolling_skewness', 'rsi', 'signal_line',
]

_handler_ma = {'s': _sma, 'w': _wma, 'e': _ema}
_handler_mstd = {'s': _smstd, 'w': _wmstd, 'e': _emstd}

# =========================================================================== #
#                                 Indicators                                  #
# =========================================================================== #


@WrapperArray('window')
def bollinger_band(
    X: NDArray,
    w: int = 20,
    n: int | float = 2,
    kind: str = 's',
    axis: int = 0,
    dtype=None,
) -> tuple[NDArray, NDArray]:
    r""" Compute the bollinger bands of size `w` for each `X`' series'.

    Bollinger Bands are a type of statistical chart characterizing the prices
    and volatility over time of a financial instrument or commodity, using a
    formulaic method propounded by J. Bollinger in the 1980s [1]_.

    Notes
    -----
    Let :math:`\mu_t` the moving average and :math:`\sigma_t` is the moving
    standard deviation of size `w` for `X` at time t.

    .. math::

        upperBand_t = \mu_t + n \times \sigma_t \\
        lowerBand_t = \mu_t - n \times \sigma_t


    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the indicator. If `X` is a two-dimensional array,
        then an indicator is computed for each series along `axis`.
    w : int, optional
        Size of the lagged window of the moving average, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is 20.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.
    n : float, optional
        Number of standard deviations above and below the moving average.
        Default is 2.
    kind : {'s', 'e', 'w'}
        - If 'e' then use exponential moving average/standard deviation, see
          :func:`~fynance.features.momentums.ema` and
          :func:`~fynance.features.momentums.emstd` for details.
        - If 's' (default) then use simple moving average/standard deviation,
          see :func:`~fynance.features.momentums.sma` and
          :func:`~fynance.features.momentums.smstd` for details.
        - If 'w' then use weighted moving average/standard deviation, see
          :func:`~fynance.features.momentums.wma` and
          :func:`~fynance.features.momentums.wmstd` for details.

    Returns
    -------
    upper_band, lower_band : np.ndarray[dtype, ndim=1 or 2]
        Respectively upper and lower bollinger bands for each series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bollinger_Bands

    Examples
    --------
    >>> import warnings
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     upper_band, lower_band = bollinger_band(X, w=3, n=2)
    >>> upper_band
    array([ 60.        , 120.        , 112.65986324, 132.65986324,
           185.31972647, 185.31972647])
    >>> lower_band
    array([60.        , 40.        , 47.34013676, 67.34013676, 54.68027353,
           54.68027353])

    See Also
    --------
    .z_score, rsi, hma, macd_hist, cci

    """
    if dtype is None:
        dtype = X.dtype

    if X.dtype != np.float64:
        X = X.astype(np.float64)

    if kind == 'e':
        w = 1 - 2 / (1 + w)  # type: ignore[assignment]

    warn(
        'Since version 1.1.0, bollinger_band returns upper and lower bands. '
        'The single-array return path is deprecated and will be removed in '
        'fynance 2.0.',
        category=DeprecationWarning,
        stacklevel=2,
    )

    if axis == 1:
        avg = _handler_ma[kind.lower()](X.T, w).T
        std = _handler_mstd[kind.lower()](X.T, w).T

    else:
        avg = _handler_ma[kind.lower()](X, w)
        std = _handler_mstd[kind.lower()](X, w)

    up_band = avg + n * std
    low_band = avg - n * std

    return (up_band).astype(dtype), (low_band).astype(dtype)


@WrapperArray('dtype', 'axis', 'window')
def cci(
    X: NDArray,
    high: NDArray | None = None,
    low: NDArray | None = None,
    w: int = 20,
    axis: int = 0,
    dtype=None,
) -> NDArray:
    r""" Compute Commodity Channel Index of size `w` for each `X`' series'.

    CCI is an oscillator introduced by Donald Lamber in 1980 [2]_. It is
    calculated as the difference between the typical price of a commodity and
    its simple moving average, divided by the moving mean absolute deviation of
    the typical price.

    Notes
    -----
    The index is usually scaled by an inverse factor of 0.015 to provide more
    readable numbers:

    .. math::

        cci = \frac{1}{0.015} \frac{p_t - sma^w_t(p)}{mad^w_t(p)} \\
        \text{where, }p = \frac{p_{close} + p_{high} + p_{low}}{3}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the indicator. If `X` is a two-dimensional array,
        then an indicator is computed for each series along `axis`.
    high, low : np.ndarray[dtype, ndim=1 or 2], optional
        Series of high and low prices, if `None` then `p_t` is computed with
        only closed prices. Must have the same shape as `X`.
    w : int, optional
        Size of the lagged window of the moving average, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is 20.
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
    .. [2] https://en.wikipedia.org/wiki/Commodity_channel_index

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> cci(X, w=3, dtype=np.float64)
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
    r_mad = roll_mad(p, w=w)
    # Avoid zero division
    r_mad[r_mad == 0.] = 1.

    return (p - _sma(p, w)) / r_mad / 0.015


@WrapperArray('dtype', 'axis', 'window')
def hma(X: NDArray, w: int = 21, kind: str = 'w', axis: int = 0, dtype=None) -> NDArray:
    r""" Compute the Hull Moving Average of size `w` for each `X`' series'.

    The Hull Moving Average, developed by A. Hull [3]_, is a financial
    indicator. It tries to reduce the lag in a moving average.

    Notes
    -----
    Let :math:`ma^w` the moving average function of lagged window size `w`.

    .. math::

        hma^w_t(X) = ma^{\sqrt{w}}_t(2 \times
        ma^{\frac{w}{2}}_t(X)) - ma^w_t(X))

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the indicator. If `X` is a two-dimensional array,
        then an indicator is computed for each series along `axis`.
    w : int, optional
        Size of the main lagged window of the moving average, must be positive.
        If ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is 21.
    kind : {'e', 's', 'w'}
        - If 'e' then use exponential moving average, see
          :func:`~fynance.features.momentums.ema` for details.
        - If 's' then use simple moving average, see
          :func:`~fynance.features.momentums.sma` for details.
        - If 'w' (default) then use weighted moving average, see
          :func:`~fynance.features.momentums.wma` for details.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Hull moving average of each series.

    References
    ----------
    .. [3] https://alanhull.com/hull-moving-average

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80])
    >>> hma(X, w=3, dtype=np.float64)
    array([ 60.        , 113.33333333,  76.66666667, 136.66666667,
           186.66666667,  46.66666667])

    See Also
    --------
    .z_score, bollinger_band, rsi, macd_hist, cci

    """
    if kind == 'e':
        w = 1 - 2 / (1 + w)  # type: ignore[assignment]

    f = _handler_ma[kind.lower()]

    ma1 = f(X, int(w / 2))
    ma2 = f(X, int(w))
    hma = f(2. * ma1 - ma2, int(np.sqrt(w)))

    return hma


@WrapperArray('dtype', 'axis')
def macd_hist(
    X: NDArray,
    w: int = 9,
    fast_w: int = 12,
    slow_w: int = 26,
    kind: str = 'e',
    axis: int = 0,
    dtype=None,
) -> NDArray:
    """ Compute Moving Average Convergence Divergence Histogram.

    MACD is a trading indicator used in technical analysis of stock prices,
    created by Gerald Appel in the late 1970s [4]_. It is designed to reveal
    changes in the strength, direction, momentum, and duration of a trend in a
    stock's price.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the indicator. If `X` is a two-dimensional array,
        then an indicator is computed for each series along `axis`.
    w : int, optional
        Size of the main lagged window of the moving average, must be positive.
        If ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is 9.
    fast_w : int, optional
        Size of the lagged window of the short moving average, must be strictly
        positive. Default is 12.
    slow_w : int, optional
        Size of the lagged window of the lond moving average, must be strictly
        positive. Default is 26.
    kind : {'e', 's', 'w'}
        - If 'e' (default) then use exponential moving average, see
          :func:`~fynance.features.momentums.ema` for details.
        - If 's' then use simple moving average, see
          :func:`~fynance.features.momentums.sma` for details.
        - If 'w' then use weighted moving average, see
          :func:`~fynance.features.momentums.wma` for details.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Moving average convergence/divergence histogram of each series.

    References
    ----------
    .. [4] https://en.wikipedia.org/wiki/MACD

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> macd_hist(X, w=3, fast_w=2, slow_w=4)
    array([ 0.        ,  5.33333333, -0.35555556,  3.93481481,  6.4102716 ,
           -9.47070947])

    See Also
    --------
    .z_score, bollinger_band, hma, macd_line, signal_line, cci

    """
    if fast_w <= 0 or slow_w <= 0:

        raise ValueError('lagged window of size {} and {} are not available, \
            must be positive.'.format(fast_w, slow_w))

    elif kind == 'e':
        w = 1 - 2 / (1 + w)  # type: ignore[assignment]

    macd_lin = _macd_line(X, fast_w, slow_w, kind)
    sig_lin = _signal_line(X, w, fast_w, slow_w, kind)

    hist = macd_lin - sig_lin

    return hist


@WrapperArray('dtype', 'axis')
def macd_line(
    X: NDArray,
    fast_w: int = 12,
    slow_w: int = 26,
    kind: str = 'e',
    axis: int = 0,
    dtype=None,
) -> NDArray:
    """ Compute Moving Average Convergence Divergence Line.

    MACD is a trading indicator used in technical analysis of stock prices,
    created by Gerald Appel in the late 1970s [4]_. It is designed to reveal
    changes in the strength, direction, momentum, and duration of a trend in a
    stock's price.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the indicator. If `X` is a two-dimensional array,
        then an indicator is computed for each series along `axis`.
    fast_w : int, optional
        Size of the lagged window of the short moving average, must be strictly
        positive. Default is 12.
    slow_w : int, optional
        Size of the lagged window of the lond moving average, must be strictly
        positive. Default is 26.
    kind : {'e', 's', 'w'}
        - If 'e' (default) then use exponential moving average, see
          :func:`~fynance.features.momentums.ema` for details.
        - If 's' then use simple moving average, see
          :func:`~fynance.features.momentums.sma` for details.
        - If 'w' then use weighted moving average, see
          :func:`~fynance.features.momentums.wma` for details.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Moving average convergence/divergence line of each series.

    References
    ----------
    .. [4] https://en.wikipedia.org/wiki/MACD

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> macd_line(X, fast_w=2, slow_w=4)
    array([ 0.        , 10.66666667,  4.62222222, 12.84740741, 21.7331358 ,
           -3.61855473])

    See Also
    --------
    .z_score, bollinger_band, hma, macd_hist, signal_line, cci

    """
    if fast_w <= 0 or slow_w <= 0:

        raise ValueError('lagged window of size {} and {} are not available, \
            must be positive.'.format(fast_w, slow_w))

    return _macd_line(X, fast_w, slow_w, kind)


def _macd_line(X, fast_w, slow_w, kind):
    if kind == 'e':
        fast_w = 1 - 2 / (fast_w + 1)
        slow_w = 1 - 2 / (slow_w + 1)

    f = _handler_ma[kind.lower()]

    fast = f(X, fast_w)
    slow = f(X, slow_w)
    macd_lin = fast - slow

    return macd_lin


@WrapperArray('dtype', 'axis', 'window')
def rsi(X: NDArray, w: int = 14, kind: str = 'e', axis: int = 0, dtype=None) -> NDArray:
    r""" Compute Relative Strenght Index.

    The relative strength index, developed by J. Welles Wilder in 1978 [5]_, is
    a technical indicator used in the analysis of financial markets. It is
    intended to chart the current and historical strength or weakness of a
    stock or market based on the closing prices of a recent trading period.

    Notes
    -----
    It is the average gain of upward periods (noted :math:`ma^w_t(X^+)`)
    divided by the average loss of downward (noted :math:`ma^w_t(X^-)`) periods
    during the specified time frame `w`, such that :

    .. math::

        RSI^w_t(X) = 100 - \frac{100}{1 + \frac{ma^w_t(X^+)}{ma^w_t(X^-)}}

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the indicator. If `X` is a two-dimensional array,
        then an indicator is computed for each series along `axis`.
    w : int, optional
        Size of the lagged window of the moving average, must be positive. If
        ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is 14.
    kind : {'e', 's', 'w'}
        - If 'e' (default) then use exponential moving average, see
          :func:`~fynance.features.momentums.ema` for details.
        - If 's' then use simple moving average, see
          :func:`~fynance.features.momentums.sma` for details.
        - If 'w' then use weighted moving average, see
          :func:`~fynance.features.momentums.wma` for details.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Relative strength index for each period.

    References
    ----------
    .. [5] https://en.wikipedia.org/wiki/Relative_strength_index

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> rsi(X, w=3)
    array([ 0.        , 99.99999804, 69.59769254, 85.55610891, 91.72201613,
           30.00294321])

    See Also
    --------
    .z_score, bollinger_band, hma, macd_hist, cci

    """
    if kind == 'e':
        w = 1 - 2 / (1 + w)  # type: ignore[assignment]

    # Compute first diff
    delta = np.log(X[1:] / X[:-1])

    # Set upward and downward arrays
    U = np.zeros(delta.shape)
    D = np.zeros(delta.shape)
    U[delta > 0] = delta[delta > 0]
    D[delta < 0] = - delta[delta < 0]

    # Compute average
    f = _handler_ma[kind.lower()]
    ma_U = f(U, w)
    ma_D = f(D, w)

    # Compute rsi values
    RSI = np.zeros(X.shape)
    RSI[1:] = 100 * ma_U / (ma_U + ma_D + 1e-8)

    return RSI


@WrapperArray('dtype', 'axis', 'window')
def signal_line(
    X: NDArray,
    w: int = 9,
    fast_w: int = 12,
    slow_w: int = 26,
    kind: str = 'e',
    axis: int = 0,
    dtype=None,
) -> NDArray:
    """ MACD Signal Line for window of size `w` with slow and fast lenght.

    MACD is a trading indicator used in technical analysis of stock prices,
    created by Gerald Appel in the late 1970s [4]_. It is designed to reveal
    changes in the strength, direction, momentum, and duration of a trend in a
    stock's price.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Elements to compute the indicator. If `X` is a two-dimensional array,
        then an indicator is computed for each series along `axis`.
    w : int, optional
        Size of the main lagged window of the moving average, must be positive.
        If ``w is None`` or ``w=0``, then ``w=X.shape[axis]``. Default is 9.
    fast_w : int, optional
        Size of the lagged window of the short moving average, must be strictly
        positive. Default is 12.
    slow_w : int, optional
        Size of the lagged window of the lond moving average, must be strictly
        positive. Default is 26.
    kind : {'e', 's', 'w'}
        - If 'e' (default) then use exponential moving average, see
          :func:`~fynance.features.momentums.ema` for details.
        - If 's' then use simple moving average, see
          :func:`~fynance.features.momentums.sma` for details.
        - If 'w' then use weighted moving average, see
          :func:`~fynance.features.momentums.wma` for details.
    axis : {0, 1}, optional
        Axis along wich the computation is done. Default is 0.
    dtype : np.dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from `X` input.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        MACD signal line of each series.

    References
    ----------
    .. [4] https://en.wikipedia.org/wiki/MACD

    Examples
    --------
    >>> X = np.array([60, 100, 80, 120, 160, 80]).astype(np.float64)
    >>> signal_line(X, w=3, fast_w=2, slow_w=4)
    array([ 0.        ,  5.33333333,  4.97777778,  8.91259259, 15.3228642 ,
            5.85215473])

    See Also
    --------
    .z_score, bollinger_band, hma, macd_hist, macd_line, cci

    """
    if fast_w <= 0 or slow_w <= 0:

        raise ValueError('lagged window of size {} and {} are not available, \
            must be positive.'.format(fast_w, slow_w))

    elif kind == 'e':
        w = 1 - 2 / (1 + w)  # type: ignore[assignment]

    return _signal_line(X, w, fast_w, slow_w, kind)


def _signal_line(X, w, fast_w, slow_w, kind):
    macd_lin = _macd_line(X, fast_w, slow_w, kind)

    f = _handler_ma[kind.lower()]

    sig_lin = f(macd_lin, w)

    return sig_lin


if __name__ == '__main__':

    import doctest

    doctest.testmod()


@WrapperArray('dtype', 'axis', 'window')
def roc(X: NDArray, w: int = 14, axis: int = 0, dtype=None) -> NDArray:
    r""" Rate of Change momentum indicator.

    Percentage change of the series over a lagged window ``w``:
    :math:`ROC^w_t = 100 \cdot (X_t / X_{t-w} - 1)`. Strictly causal —
    the first ``w`` points use the first observation as the base (expanding).

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Price/level series.
    w : int, optional
        Lag window. Default 14.
    axis : {0, 1}, optional
        Axis of computation. Default 0.
    dtype : np.dtype, optional
        Output dtype.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Rate of change in percent.

    Examples
    --------
    >>> X = np.array([10., 11., 12., 13., 14.])
    >>> roc(X, w=2)
    array([ 0.        , 10.        , 20.        , 18.18181818, 16.66666667])

    """
    out = np.zeros(X.shape)
    out[1:w] = (X[1:w] / X[0] - 1.) * 100.
    out[w:] = (X[w:] / X[:-w] - 1.) * 100.

    return out


@WrapperArray('dtype', 'axis', 'window')
def realized_volatility(X: NDArray, w: int = 21, period: int = 252, axis: int = 0, dtype=None) -> NDArray:
    r""" Annualized realized volatility from a price series.

    Rolling standard deviation of log-returns, annualized by
    :math:`\sqrt{period}`. Causal: returns are strictly past, the window
    expands over the first observations.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Price/level series.
    w : int, optional
        Rolling window over returns. Default 21.
    period : int, optional
        Annualization factor (e.g. 252 daily). Default 252.
    axis : {0, 1}, optional
        Axis of computation. Default 0.
    dtype : np.dtype, optional
        Output dtype.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Annualized realized volatility, aligned to ``X`` (first entry 0).

    """
    r = np.log(X[1:] / X[:-1])
    std = np.asarray(_smstd(r, w))
    out = np.zeros(X.shape)
    out[1:] = np.sqrt(period) * std

    return out


def _rolling_apply(X, w, fn, min_obs):
    # Causal rolling reduction: window [max(0, t-w+1), t], expanding at start.
    out = np.zeros(X.shape)
    n = X.shape[0]
    for t in range(n):
        lo = max(0, t - w + 1)
        win = X[lo:t + 1]
        if win.shape[0] >= min_obs:
            out[t] = fn(win)

    return out


@WrapperArray('dtype', 'axis', 'window')
def rolling_skewness(X: NDArray, w: int = 21, axis: int = 0, dtype=None) -> NDArray:
    """ Rolling sample skewness over a strictly-past window ``w``.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Input series.
    w : int, optional
        Rolling window. Default 21.
    axis : {0, 1}, optional
        Axis of computation. Default 0.
    dtype : np.dtype, optional
        Output dtype.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Rolling skewness (0 where fewer than 3 observations are available).

    """
    return np.nan_to_num(
        _rolling_apply(X, w, lambda win: _sp_stats.skew(win, axis=0), min_obs=3)
    )


@WrapperArray('dtype', 'axis', 'window')
def rolling_kurtosis(X: NDArray, w: int = 21, axis: int = 0, dtype=None) -> NDArray:
    """ Rolling excess kurtosis over a strictly-past window ``w``.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Input series.
    w : int, optional
        Rolling window. Default 21.
    axis : {0, 1}, optional
        Axis of computation. Default 0.
    dtype : np.dtype, optional
        Output dtype.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Rolling excess kurtosis (0 where fewer than 4 observations).

    """
    return np.nan_to_num(
        _rolling_apply(X, w, lambda win: _sp_stats.kurtosis(win, axis=0), min_obs=4)
    )


@WrapperArray('dtype', 'axis', 'window')
def rolling_autocorr(X: NDArray, w: int = 21, lag: int = 1, axis: int = 0, dtype=None) -> NDArray:
    """ Rolling lag-``lag`` autocorrelation over a strictly-past window ``w``.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Input series.
    w : int, optional
        Rolling window. Default 21.
    lag : int, optional
        Autocorrelation lag. Default 1.
    axis : {0, 1}, optional
        Axis of computation. Default 0.
    dtype : np.dtype, optional
        Output dtype.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Rolling autocorrelation (0 where the window is too short).

    """
    def _ac(win):
        a, b = win[:-lag], win[lag:]
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(denom > 0, (a * b).sum(axis=0) / denom, 0.)

    return np.nan_to_num(_rolling_apply(X, w, _ac, min_obs=lag + 2))
