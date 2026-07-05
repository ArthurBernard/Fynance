#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-18 21:15:59
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-18 22:21:22

""" Rolling extremum and pairwise statistics.

Rolling minimum and maximum over a lagged window. Used directly as
features (e.g. price-channel breakouts) and internally by
:func:`fynance.features.scale.roll_normalize` to compute lookahead-safe
min-max scaling parameters.

Also provides trailing *pairwise* rolling statistics between two aligned
1-D series (covariance, Pearson correlation, OLS beta) and a full-sample
lead-lag cross-correlation profile — e.g. a rolling hedge ratio
(:func:`roll_beta`) or checking whether one series leads another
(:func:`cross_corr`).

Main entry points
-----------------
- :func:`roll_min` — rolling minimum over a window.
- :func:`roll_max` — rolling maximum over a window.
- :func:`roll_cov` — trailing rolling covariance between two series.
- :func:`roll_corr` — trailing rolling Pearson correlation between two series.
- :func:`roll_beta` — trailing rolling OLS slope of ``x`` on ``y``.
- :func:`cross_corr` — full-sample lead-lag cross-correlation profile.

"""

from __future__ import annotations

# Built-in packages
# Third party packages
import numpy as np

# Local packages
from numba import njit, prange
from numpy.typing import NDArray

from fynance._wrappers import WrapperArray

__all__ = ["roll_min", "roll_max", "roll_cov", "roll_corr", "roll_beta", "cross_corr"]


@njit(cache=True)
def _roll_min_1d(X, w):
    """ Rolling minimum over a trailing window of size ``w``.

    O(n) monotonic-deque implementation (each index is pushed/popped once);
    the returned minima are identical to the naive O(n*w) computation.
    """
    T = X.shape[0]
    out = np.empty(T, dtype=np.float64)
    dq = np.empty(T, dtype=np.int64)  # indices, values increasing front->back
    head = 0
    tail = 0
    for t in range(T):
        while tail > head and X[dq[tail - 1]] >= X[t]:
            tail -= 1
        dq[tail] = t
        tail += 1
        if dq[head] < t - w + 1:
            head += 1
        out[t] = X[dq[head]]
    return out


@njit(parallel=True, cache=True)
def _roll_min_2d(X, w):
    """ Column-wise rolling minimum (O(n) per column, parallel over columns). """
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in prange(N):
        out[:, n] = _roll_min_1d(np.ascontiguousarray(X[:, n]), w)
    return out


@njit(cache=True)
def _roll_max_1d(X, w):
    """ Rolling maximum over a trailing window of size ``w``.

    O(n) monotonic-deque implementation; identical maxima to the naive form.
    """
    T = X.shape[0]
    out = np.empty(T, dtype=np.float64)
    dq = np.empty(T, dtype=np.int64)  # indices, values decreasing front->back
    head = 0
    tail = 0
    for t in range(T):
        while tail > head and X[dq[tail - 1]] <= X[t]:
            tail -= 1
        dq[tail] = t
        tail += 1
        if dq[head] < t - w + 1:
            head += 1
        out[t] = X[dq[head]]
    return out


@njit(parallel=True, cache=True)
def _roll_max_2d(X, w):
    """ Column-wise rolling maximum (O(n) per column, parallel over columns). """
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for n in prange(N):
        out[:, n] = _roll_max_1d(np.ascontiguousarray(X[:, n]), w)
    return out



# =========================================================================== #
#                                   Min Max                                   #
# =========================================================================== #


@WrapperArray('dtype', 'axis', 'window')
def roll_min(X: NDArray, w: int | None = None, axis: int = 0, dtype=None) -> NDArray:
    r""" Compute simple rolling minimum of size `w` for each `X`' series.

    .. math::

        roll\_min^w_t(X) = min(X_{t - w + 1}, ..., X_t)

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

        roll\_max^w_t(X) = max(X_{t - w + 1}, ..., X_t)

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
    roll_min

    """
    return _roll_max(X, w)


def _roll_max(X, w):
    if len(X.shape) == 2:

        return _roll_max_2d(X, w)

    return _roll_max_1d(X, w)


# =========================================================================== #
#                          Pairwise rolling statistics                        #
# =========================================================================== #


def _validate_pair(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """ Cast ``x``/``y`` to contiguous float64 1-D arrays and validate them. """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))

    if x.ndim != 1 or y.ndim != 1:

        raise ValueError(
            f"x and y must be 1-D, got ndim={x.ndim} and ndim={y.ndim}"
        )

    if x.shape[0] != y.shape[0]:

        raise ValueError(
            f"x and y must have the same length, got {x.shape[0]} and "
            f"{y.shape[0]}"
        )

    if np.isnan(x).any() or np.isnan(y).any():

        raise ValueError("x and y must not contain NaN")

    return x, y


def _validate_window(w: int) -> int:
    """ Validate the trailing window size ``w`` (must be an int >= 2). """
    if not isinstance(w, (int, np.integer)) or w < 2:

        raise ValueError(f"w must be an integer >= 2, got {w!r}")

    return int(w)


@njit(cache=True)
def _roll_cov_1d(x, y, w):
    """ Trailing rolling covariance (ddof=0) of two aligned 1-D series. """
    T = x.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        if t < w - 1:
            out[t] = np.nan
            continue

        start = t - w + 1
        sx = 0.0
        sy = 0.0
        for i in range(start, t + 1):
            sx += x[i]
            sy += y[i]
        mx = sx / w
        my = sy / w

        sxy = 0.0
        for i in range(start, t + 1):
            sxy += (x[i] - mx) * (y[i] - my)
        out[t] = sxy / w

    return out


@njit(cache=True)
def _roll_corr_1d(x, y, w):
    """ Trailing rolling Pearson correlation (ddof cancels) of two series. """
    T = x.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        if t < w - 1:
            out[t] = np.nan
            continue

        start = t - w + 1
        sx = 0.0
        sy = 0.0
        for i in range(start, t + 1):
            sx += x[i]
            sy += y[i]
        mx = sx / w
        my = sy / w

        sxy = 0.0
        sxx = 0.0
        syy = 0.0
        for i in range(start, t + 1):
            dx = x[i] - mx
            dy = y[i] - my
            sxy += dx * dy
            sxx += dx * dx
            syy += dy * dy

        denom = np.sqrt(sxx * syy)
        if denom == 0.0:
            out[t] = np.nan
        else:
            out[t] = sxy / denom

    return out


@njit(cache=True)
def _roll_beta_1d(x, y, w):
    """ Trailing rolling OLS slope of ``x`` on ``y`` (ddof cancels). """
    T = x.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        if t < w - 1:
            out[t] = np.nan
            continue

        start = t - w + 1
        sx = 0.0
        sy = 0.0
        for i in range(start, t + 1):
            sx += x[i]
            sy += y[i]
        mx = sx / w
        my = sy / w

        sxy = 0.0
        syy = 0.0
        for i in range(start, t + 1):
            dx = x[i] - mx
            dy = y[i] - my
            sxy += dx * dy
            syy += dy * dy

        if syy == 0.0:
            out[t] = np.nan
        else:
            out[t] = sxy / syy

    return out


@njit(cache=True)
def _cross_corr_1d(x, y, max_lag):
    """ Full-sample corr(x[t], y[t - lag]) for lag in [-max_lag, max_lag]. """
    T = x.shape[0]
    n_lags = 2 * max_lag + 1
    out = np.empty(n_lags, dtype=np.float64)

    for idx in range(n_lags):
        lag = idx - max_lag

        if lag >= 0:
            n = T - lag
            sx = 0.0
            sy = 0.0
            for i in range(n):
                sx += x[lag + i]
                sy += y[i]
            mx = sx / n
            my = sy / n

            sxy = 0.0
            sxx = 0.0
            syy = 0.0
            for i in range(n):
                dx = x[lag + i] - mx
                dy = y[i] - my
                sxy += dx * dy
                sxx += dx * dx
                syy += dy * dy
        else:
            k = -lag
            n = T - k
            sx = 0.0
            sy = 0.0
            for i in range(n):
                sx += x[i]
                sy += y[k + i]
            mx = sx / n
            my = sy / n

            sxy = 0.0
            sxx = 0.0
            syy = 0.0
            for i in range(n):
                dx = x[i] - mx
                dy = y[k + i] - my
                sxy += dx * dy
                sxx += dx * dx
                syy += dy * dy

        denom = np.sqrt(sxx * syy)
        if denom == 0.0:
            out[idx] = np.nan
        else:
            out[idx] = sxy / denom

    return out


def roll_cov(x: NDArray, y: NDArray, w: int = 63) -> NDArray:
    r""" Trailing rolling covariance between two aligned 1-D series.

    .. math::

        roll\_cov^w_t(x, y) = \frac{1}{w} \sum_{i=t-w+1}^{t}
            (x_i - \bar{x}_t)(y_i - \bar{y}_t)

    where :math:`\bar{x}_t` and :math:`\bar{y}_t` are the means of ``x`` and
    ``y`` over the trailing window :math:`[t - w + 1, t]` (inclusive of
    ``t``, the "house" trailing-window convention shared with
    :func:`roll_min`/:func:`roll_max`). The variance used is the *biased*
    (``ddof=0``) estimator.

    Parameters
    ----------
    x : np.ndarray[float64, ndim=1]
        First series. Cast to ``float64`` if needed; must not contain NaN.
    y : np.ndarray[float64, ndim=1]
        Second series, same length as ``x``. Cast to ``float64`` if needed;
        must not contain NaN.
    w : int, optional
        Size of the trailing window, must be an integer >= 2. Default is 63.

    Returns
    -------
    np.ndarray[float64, ndim=1]
        Trailing rolling covariance. The first ``w - 1`` entries are
        ``np.nan`` (insufficient history).

    Examples
    --------
    >>> x = np.array([1., 2., 3., 4., 5., 6.])
    >>> y = 2 * x
    >>> roll_cov(x, y, w=2)
    array([nan, 0.5, 0.5, 0.5, 0.5, 0.5])

    See Also
    --------
    roll_corr, roll_beta

    """
    x, y = _validate_pair(x, y)
    w = _validate_window(w)

    return _roll_cov_1d(x, y, w)


def roll_corr(x: NDArray, y: NDArray, w: int = 63) -> NDArray:
    r""" Trailing rolling Pearson correlation between two aligned 1-D series.

    .. math::

        roll\_corr^w_t(x, y) = \frac{
            \sum_{i=t-w+1}^{t} (x_i - \bar{x}_t)(y_i - \bar{y}_t)
        }{
            \sqrt{\sum_{i=t-w+1}^{t} (x_i - \bar{x}_t)^2}
            \sqrt{\sum_{i=t-w+1}^{t} (y_i - \bar{y}_t)^2}
        }

    over the trailing window :math:`[t - w + 1, t]` (inclusive of ``t``, see
    :func:`roll_cov`). The ``1/w`` normalization of the covariance and the two
    variances cancels out, so the result does not depend on ``ddof``.

    Parameters
    ----------
    x : np.ndarray[float64, ndim=1]
        First series. Cast to ``float64`` if needed; must not contain NaN.
    y : np.ndarray[float64, ndim=1]
        Second series, same length as ``x``. Cast to ``float64`` if needed;
        must not contain NaN.
    w : int, optional
        Size of the trailing window, must be an integer >= 2. Default is 63.

    Returns
    -------
    np.ndarray[float64, ndim=1]
        Trailing rolling correlation, in ``[-1, 1]``. The first ``w - 1``
        entries are ``np.nan`` (insufficient history). If either series has
        zero variance within a window, that entry is ``np.nan`` (checked
        explicitly before dividing — no ``RuntimeWarning`` is raised).

    Examples
    --------
    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> y = 2 * x
    >>> roll_corr(x, y, w=3)
    array([nan, nan,  1.,  1.,  1.])

    See Also
    --------
    roll_cov, roll_beta

    """
    x, y = _validate_pair(x, y)
    w = _validate_window(w)

    return _roll_corr_1d(x, y, w)


def roll_beta(x: NDArray, y: NDArray, w: int = 63) -> NDArray:
    r""" Trailing rolling OLS slope of ``x`` regressed on ``y``.

    .. math::

        roll\_beta^w_t(x, y) = \frac{roll\_cov^w_t(x, y)}{roll\_var^w_t(y)}

    i.e. the slope of the univariate OLS regression of ``x`` on ``y`` over
    the trailing window :math:`[t - w + 1, t]` (inclusive of ``t``, see
    :func:`roll_cov`). Typical use is a rolling hedge ratio: how many units
    of ``y`` are needed to hedge one unit of ``x``. As with :func:`roll_corr`,
    the ``1/w`` normalization cancels, so the result does not depend on
    ``ddof``.

    Parameters
    ----------
    x : np.ndarray[float64, ndim=1]
        Dependent series. Cast to ``float64`` if needed; must not contain
        NaN.
    y : np.ndarray[float64, ndim=1]
        Independent (regressor) series, same length as ``x``. Cast to
        ``float64`` if needed; must not contain NaN.
    w : int, optional
        Size of the trailing window, must be an integer >= 2. Default is 63.

    Returns
    -------
    np.ndarray[float64, ndim=1]
        Trailing rolling OLS slope. The first ``w - 1`` entries are
        ``np.nan`` (insufficient history). If ``y`` has zero variance within
        a window, that entry is ``np.nan`` (checked explicitly before
        dividing — no ``RuntimeWarning`` is raised).

    Examples
    --------
    ``y = 2 * x`` has a constant beta of 0.5 (``cov(x, y) / var(y) =
    2 var(x) / 4 var(x)``), regardless of the window content:

    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> y = 2 * x
    >>> roll_beta(x, y, w=3)
    array([nan, nan, 0.5, 0.5, 0.5])

    See Also
    --------
    roll_cov, roll_corr

    """
    x, y = _validate_pair(x, y)
    w = _validate_window(w)

    return _roll_beta_1d(x, y, w)


def cross_corr(x: NDArray, y: NDArray, max_lag: int = 20) -> NDArray:
    r""" Full-sample lead-lag cross-correlation profile of two series.

    .. math::

        cross\_corr_{lag}(x, y) = corr(x_t, y_{t - lag}),
        \quad lag \in \{-max\_lag, \dots, max\_lag\}

    where the correlation is computed over the full overlapping sample (all
    valid ``t``, i.e. ``T - |lag|`` pairs), *not* a trailing window.

    **Lag convention.** Entry ``lag`` correlates ``x[t]`` with ``y[t - lag]``.
    A positive ``lag`` therefore pairs ``x`` at time ``t`` with ``y`` *earlier*
    in the series (at ``t - lag``): if that pairing is where the correlation
    peaks, ``y``'s past values line up with ``x``'s current values, i.e.
    ``y`` **leads** ``x`` by ``lag`` bars. Conversely a negative ``lag`` that
    maximizes the correlation means ``y`` **lags** ``x`` (``x`` leads ``y``)
    by ``|lag|`` bars. For example, if ``y[t] = x[t - 3]`` (``y`` is ``x``
    delayed by 3 bars, so ``x`` leads ``y`` by 3), the profile peaks at
    ``lag = -3``, since ``y[t - (-3)] = y[t + 3] = x[t]``.

    Parameters
    ----------
    x : np.ndarray[float64, ndim=1]
        First series. Cast to ``float64`` if needed; must not contain NaN.
    y : np.ndarray[float64, ndim=1]
        Second series, same length as ``x``. Cast to ``float64`` if needed;
        must not contain NaN.
    max_lag : int, optional
        Maximum absolute lag to scan, must be a non-negative integer and
        strictly less than ``len(x)``. Default is 20.

    Returns
    -------
    np.ndarray[float64, ndim=1]
        Cross-correlation profile of length ``2 * max_lag + 1``, ordered from
        ``lag=-max_lag`` to ``lag=max_lag``. Entries where either side has
        zero variance over the overlap are ``np.nan`` (checked explicitly
        before dividing — no ``RuntimeWarning`` is raised).

    Examples
    --------
    ``y`` is ``x`` delayed by 3 bars (``x`` leads ``y`` by 3): the profile
    peaks at ``lag = -3``.

    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=200)
    >>> y = np.roll(x, 3)
    >>> y[:3] = rng.normal(size=3)  # avoid a spurious wrap-around match
    >>> profile = cross_corr(x, y, max_lag=5)
    >>> profile.shape
    (11,)
    >>> int(np.arange(-5, 6)[np.argmax(profile)])
    -3

    See Also
    --------
    roll_corr

    """
    x, y = _validate_pair(x, y)

    if not isinstance(max_lag, (int, np.integer)) or max_lag < 0:

        raise ValueError(
            f"max_lag must be a non-negative integer, got {max_lag!r}"
        )

    if max_lag >= x.shape[0]:

        raise ValueError(
            f"max_lag={max_lag} must be strictly less than len(x)={x.shape[0]}"
        )

    return _cross_corr_1d(x, y, int(max_lag))
