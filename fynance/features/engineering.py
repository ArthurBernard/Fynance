#!/usr/bin/env python3
# coding: utf-8

""" Feature-engineering & selection research tools.

Multi-resolution feature stacking, incremental (O(1)) moment updates, and a
Granger-causality test for filtering candidate features.
"""

from __future__ import annotations

# Built-in packages
from typing import Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray
from scipy import stats as _sp_stats

__all__ = ['IncrementalMoments', 'granger_causality', 'multi_resolution']


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
