#!/usr/bin/env python3
# coding: utf-8

""" Feature-engineering & selection research tools.

Multi-resolution feature stacking, incremental (O(1)) moment updates, a
Granger-causality test for filtering candidate features, and fixed-width
fractional differentiation (:func:`fracdiff`) for stationarizing a series
while preserving memory.
"""

from __future__ import annotations

# Built-in packages
from collections.abc import Mapping
from typing import Any, Callable

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import stats as _sp_stats

# Local packages
from fynance.features.indicators import realized_volatility

__all__ = [
    'IncrementalMoments', 'adaptive_roll', 'adaptive_volatility',
    'fracdiff', 'granger_causality', 'multi_resolution',
]


def multi_resolution(
    func: Callable[..., NDArray], X: NDArray, windows, **kwargs,
) -> NDArray:
    r""" Stack a window-based feature computed at several resolutions.

    Applies ``func(X, w, **kwargs)`` for every window ``w`` in ``windows`` and
    column-stacks the results, letting a model learn the relevant horizon
    instead of fixing one.

    Parameters
    ----------
    func : callable
        A feature function taking ``(X, w, **kwargs)`` (e.g.
        :func:`~fynance.features.momentums.sma`,
        :func:`~fynance.features.indicators.realized_volatility`).
    X : np.ndarray
        One-dimensional input series.
    windows : iterable of int
        Window sizes / resolutions.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(X), len(windows))``, one column per resolution.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.features.momentums import sma
    >>> X = np.arange(1., 6.)
    >>> multi_resolution(sma, X, [2, 3]).shape
    (5, 2)

    """
    cols = [np.asarray(func(X, w, **kwargs)).reshape(-1) for w in windows]

    return np.column_stack(cols)


def granger_causality(x: NDArray, y: NDArray, lag: int = 1) -> tuple[float, float]:
    r""" Granger-causality F-test: does ``x`` help predict ``y``?

    Compares a restricted autoregression of ``y`` on its own lags with an
    unrestricted one that also includes lags of ``x``. A small p-value means
    ``x`` Granger-causes ``y`` (adds predictive power beyond ``y``'s past).

    Parameters
    ----------
    x, y : np.ndarray
        One-dimensional series of equal length.
    lag : int, optional
        Number of lags. Default 1.

    Returns
    -------
    f_stat : float
        F statistic of the restricted-vs-unrestricted comparison.
    p_value : float
        Associated p-value (low → ``x`` Granger-causes ``y``).

    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.shape[0] - lag
    if n <= 2 * lag + 1:
        raise ValueError("series too short for the requested lag")

    target = y[lag:]
    y_lags = np.column_stack([y[lag - k - 1:-k - 1] for k in range(lag)])
    x_lags = np.column_stack([x[lag - k - 1:-k - 1] for k in range(lag)])
    ones = np.ones((n, 1))

    def _rss(design):
        beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ beta
        return float(resid @ resid)

    rss_r = _rss(np.hstack([ones, y_lags]))
    rss_u = _rss(np.hstack([ones, y_lags, x_lags]))

    df_u = n - (2 * lag + 1)
    f_stat = ((rss_r - rss_u) / lag) / (rss_u / df_u + 1e-12)
    p_value = float(_sp_stats.f.sf(f_stat, lag, df_u))

    return float(f_stat), p_value


class IncrementalMoments:
    """ Online mean / variance via Welford's algorithm (O(1) per update).

    Streaming alternative to recomputing a rolling mean/variance from scratch.

    Attributes
    ----------
    n : int
        Number of observations seen.
    mean : float
        Running mean.

    Examples
    --------
    >>> im = IncrementalMoments()
    >>> for v in [1.0, 2.0, 3.0]:
    ...     _ = im.update(v)
    >>> im.mean, round(im.var, 4)
    (2.0, 0.6667)

    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> "IncrementalMoments":
        """ Incorporate one observation; return self for chaining. """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

        return self

    @property
    def var(self) -> float:
        """ Population variance (0 before the second observation). """
        return self._m2 / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        """ Population standard deviation. """
        return self.var ** 0.5


def adaptive_roll(
    X: NDArray,
    func: Callable[..., NDArray],
    windows: Mapping[int, int],
    regimes: NDArray,
    **kwargs: Any,
) -> NDArray:
    r""" Apply a window-based feature with a **regime-dependent** window.

    At each bar ``t`` the output is ``func(X, windows[regimes[t]])[t]`` — a short
    window in one regime, a longer one in another. Causal as long as both inputs
    are: ``func`` must be a trailing-window feature (value at ``t`` uses
    ``X[..t]``) and ``regimes`` a causal label (e.g. from
    :class:`~fynance.features.RegimeDetector`, fit-on-train / assign-online).

    Parameters
    ----------
    X : np.ndarray
        One-dimensional input series.
    func : callable
        A trailing-window feature taking ``(X, w, **kwargs)`` and returning an
        array aligned with ``X`` (e.g.
        :func:`~fynance.features.momentums.sma`,
        :func:`~fynance.features.indicators.realized_volatility`).
    windows : mapping of int to int
        Window size for each regime label. Must cover every label present in
        ``regimes``.
    regimes : np.ndarray
        Causal integer regime label per bar, aligned with ``X``.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    np.ndarray
        The regime-adaptive feature, shape ``(len(X),)``.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.features.momentums import sma
    >>> X = np.arange(1., 7.)
    >>> regimes = np.array([0, 0, 0, 1, 1, 1])
    >>> adaptive_roll(X, sma, {0: 1, 1: 3}, regimes)
    array([1., 2., 3., 3., 4., 5.])

    """
    x = np.asarray(X, dtype=np.float64).reshape(-1)
    reg = np.asarray(regimes).reshape(-1)

    if reg.size != x.size:

        raise ValueError(
            f"regimes length {reg.size} != X length {x.size}"
        )

    present = set(int(r) for r in np.unique(reg))
    missing = present - set(windows)

    if missing:

        raise ValueError(f"windows has no entry for regime(s) {sorted(missing)}")

    # Compute the feature once per distinct window, then select per bar.
    out = np.empty(x.size, dtype=np.float64)
    for w in set(windows.values()):
        col = np.asarray(func(x, w, **kwargs)).reshape(-1)
        labels_with_w = [lab for lab, win in windows.items() if win == w]
        mask = np.isin(reg, labels_with_w)
        out[mask] = col[mask]

    return out


def adaptive_volatility(
    X: NDArray,
    windows: Mapping[int, int],
    regimes: NDArray,
    period: int = 252,
) -> NDArray:
    r""" Regime-adaptive realized volatility (worked example of :func:`adaptive_roll`).

    Uses a short volatility window in some regimes and a longer one in others, so
    the estimate reacts fast in turbulent regimes and stays smooth in calm ones.

    Parameters
    ----------
    X : np.ndarray
        One-dimensional price/level series.
    windows : mapping of int to int
        Volatility window for each regime label.
    regimes : np.ndarray
        Causal integer regime label per bar, aligned with ``X``.
    period : int, optional
        Annualization factor. Default 252.

    Returns
    -------
    np.ndarray
        Regime-adaptive annualized volatility, shape ``(len(X),)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = 100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.01))
    >>> regimes = (np.arange(100) // 50)   # two regimes
    >>> adaptive_volatility(X, {0: 5, 1: 20}, regimes).shape
    (100,)

    """
    return adaptive_roll(
        X, realized_volatility, windows, regimes, period=period,
    )


def _fracdiff_weights(d: float, tol: float) -> NDArray:
    """ Fixed-width weights of the fractional-differentiation operator.

    Generates ``w_0, w_1, ..., w_{K-1}`` from the binomial-series recursion
    ``w_0 = 1``, ``w_k = -w_{k-1} * (d - k + 1) / k``, stopping (and
    discarding) the first weight whose magnitude drops below ``tol`` — this
    is the fixed-width-window truncation of Lopez de Prado (2018), ch. 5.
    At least ``w_0`` is always kept.

    Parameters
    ----------
    d : float
        Order of differentiation.
    tol : float
        Weight-magnitude cutoff below which the (infinite, in general) series
        of weights is truncated.

    Returns
    -------
    np.ndarray
        Weights ``[w_0, ..., w_{K-1}]``, shape ``(K,)``.

    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < tol:
            break
        weights.append(w_k)
        k += 1

    return np.asarray(weights, dtype=np.float64)


@njit(cache=True)
def _fracdiff_kernel(X, w):
    T = X.shape[0]
    K = w.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        if t < K - 1:
            out[t] = np.nan
        else:
            s = 0.0
            for k in range(K):
                s += w[k] * X[t - k]
            out[t] = s

    return out


def fracdiff(X: NDArray, d: float = 0.4, tol: float = 1e-5) -> NDArray:
    r""" Fixed-width-window fractional differentiation of a price series.

    Stationarizes a (typically non-stationary, e.g. price or log-price)
    series while retaining as much memory as possible, unlike integer
    differencing (``d=1``) which is stationary but wipes out most of the
    long-run dependence. Applies the fractional difference operator
    :math:`(1-L)^d` — where :math:`L` is the lag operator — truncated to a
    fixed-width window, so that it is usable causally (online) rather than
    needing the full history at every step as the "expanding window"
    variant does.

    The weights follow the binomial-series recursion

    .. math::

        w_0 = 1, \qquad w_k = -w_{k-1} \frac{d - k + 1}{k},

    truncated to the first :math:`K` terms such that :math:`|w_K| < tol`
    (:math:`K` is fixed for the whole series — "fixed-width window"). The
    output is the causal convolution

    .. math::

        y_t = \sum_{k=0}^{K-1} w_k X_{t-k}, \qquad t \ge K - 1,

    with the first :math:`K - 1` entries set to NaN (insufficient history).
    Only past and current values of ``X`` are used, so :func:`fracdiff` is
    strictly causal and safe to use in a walk-forward / online setting.

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1 or 2]
        Input series (e.g. price level). If two-dimensional, shape
        ``(T, N)``, each column is treated independently. Must be finite
        (no NaN / inf).
    d : float, optional
        Order of differentiation, must lie in ``[0, 2]``. ``d=0`` leaves the
        series unchanged (post-warmup); ``d=1`` reduces, with the default
        `tol`, to the ordinary first difference; non-integer ``d`` in
        between trades off memory (small ``d``) against stationarity (large
        ``d``). Default is 0.4.
    tol : float, optional
        Weight-magnitude cutoff used to fix the window width :math:`K` (see
        :func:`_fracdiff_weights`). Smaller `tol` keeps more weights (longer
        memory, larger warmup) at the cost of more computation. Default is
        1e-5.

    Returns
    -------
    np.ndarray[float64, ndim=1 or 2]
        Fractionally differentiated series, same shape as `X`. The first
        ``K - 1`` rows are NaN. If ``X`` has fewer than ``K`` observations,
        the output is entirely NaN.

    Raises
    ------
    ValueError
        If `d` is not in ``[0, 2]``, if `X` contains non-finite values, or if
        `X` is not 1-D or 2-D.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    >>> fracdiff(X, d=1.0)
    array([nan,  1.,  2.,  3.,  4.])
    >>> np.array_equal(fracdiff(X, d=1.0)[1:], np.diff(X))
    True
    >>> fracdiff(X, d=0.0)
    array([ 1.,  2.,  4.,  7., 11.])

    Notes
    -----
    There is an inherent memory-vs-stationarity trade-off (Lopez de Prado,
    2018, ch. 5): larger ``d`` differentiates more aggressively, making the
    series more likely to be stationary (e.g. pass an ADF test) but erasing
    more of the long-run memory that predictive models rely on; smaller
    ``d`` preserves memory but may leave the series non-stationary. The
    common recipe is to search for the minimal ``d`` for which the
    fractionally differentiated series is stationary.

    References
    ----------
    M. Lopez de Prado, "Advances in Financial Machine Learning", Wiley,
    2018, ch. 5.

    See Also
    --------
    multi_resolution, adaptive_roll

    """
    if not (0 <= d <= 2):

        raise ValueError(f"d must be in [0, 2], got {d}")

    x = np.asarray(X, dtype=np.float64)

    if not np.all(np.isfinite(x)):

        raise ValueError("X must contain only finite values (no NaN/inf)")

    w = _fracdiff_weights(d, tol)

    if x.ndim == 1:

        return _fracdiff_kernel(x, w)

    elif x.ndim == 2:

        cols = [_fracdiff_kernel(np.ascontiguousarray(x[:, j]), w)
                for j in range(x.shape[1])]

        return np.column_stack(cols)

    else:

        raise ValueError(f"X must be 1-D or 2-D, got ndim={x.ndim}")
