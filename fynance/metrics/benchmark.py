#!/usr/bin/env python3
# coding: utf-8

""" Benchmark-relative performance metrics.

Unlike the single-curve ratios in :mod:`fynance.metrics.ratios` (Sharpe,
Sortino, Calmar, ...), the metrics here score a strategy *against* a benchmark:
beta and Jensen's alpha decompose the strategy's return into a benchmark-driven
part and a residual, while tracking error, the information ratio and the
up/down capture ratios describe how the active (strategy-minus-benchmark)
return behaves.

**Input convention.** Like :mod:`fynance.metrics.ratios`, every function here
takes two aligned 1-D **price/level curves** ``X`` (strategy) and ``B``
(benchmark) — not returns — and derives returns internally. Following
:func:`~fynance.metrics.ratios.sharpe`/:func:`~fynance.metrics.ratios.sortino`
(rather than :func:`~fynance.metrics.ratios.annual_volatility`), returns are
**simple** (arithmetic), not logarithmic: ``R_1 = 0`` and
``R_{2:T} = X_{2:T} / X_{1:T-1} - 1``. ``X`` and ``B`` must be 1-D and of the
same length (:class:`ValueError` otherwise).

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.features._metrics_helpers import _compute_returns
from fynance.features.roll_functions import roll_beta
from fynance.metrics.ratios import _safe_ratio

__all__ = [
    'beta',
    'alpha',
    'tracking_error',
    'information_ratio',
    'capture_ratio',
    'benchmark_summary',
    'roll_beta_benchmark',
]


def _validate_curves(X: NDArray, B: NDArray) -> tuple[NDArray, NDArray]:
    """ Cast ``X``/``B`` to 1-D float64 arrays and validate their shape. """
    X = np.asarray(X, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if X.ndim != 1 or B.ndim != 1:

        raise ValueError(
            f"X and B must be 1-D, got ndim={X.ndim} and ndim={B.ndim}"
        )

    if X.shape[0] != B.shape[0]:

        raise ValueError(
            f"X and B must have the same length, got {X.shape[0]} and "
            f"{B.shape[0]}"
        )

    return X, B


def _active_returns(X: NDArray, B: NDArray) -> tuple[NDArray, NDArray]:
    """ Simple returns of ``X`` and ``B`` (``R_1 = 0``, see module docstring). """
    return _compute_returns(X, False), _compute_returns(B, False)


def _ols_beta(x_ret: NDArray, b_ret: NDArray) -> float:
    """ OLS slope of ``x_ret`` on ``b_ret`` (population cov / population var). """
    dx = x_ret - x_ret.mean()
    db = b_ret - b_ret.mean()
    cov = np.asarray(np.mean(dx * db))
    var = np.asarray(np.mean(db * db))

    return float(_safe_ratio(cov, var))


def _ann_geo_return(r: NDArray, period: int) -> float:
    """ Annualized compounded (geometric) return of a return sample ``r``.

    Returns ``np.nan`` when ``r`` is empty (no bar to compound over).
    """
    n = r.shape[0]

    if n == 0:

        return float('nan')

    compounded = np.prod(1.0 + r)

    return float(compounded ** (period / n) - 1.0)


def beta(X: NDArray, B: NDArray, period: int = 252) -> float:
    r""" OLS slope of the strategy's returns on the benchmark's returns.

    Notes
    -----
    With :math:`x` the strategy's simple returns and :math:`b` the
    benchmark's simple returns (see module docstring):

    .. math::

        \beta = \frac{Cov(x, b)}{Var(b)}

    ``period`` does not affect :math:`\beta` (a slope is scale-free); it is
    accepted for signature consistency with the rest of this module (e.g.
    :func:`alpha` calls :func:`beta` internally).

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1]
        Strategy price/level curve.
    B : np.ndarray[float64, ndim=1]
        Benchmark price/level curve, same length as ``X``.
    period : int, optional
        Unused by ``beta`` itself; accepted for API consistency. Default 252.

    Returns
    -------
    float
        OLS beta of the strategy against the benchmark. ``0`` when the
        benchmark has zero variance and the strategy's covariance with it is
        also zero, ``+inf``/``-inf`` when the benchmark has zero variance and
        the covariance is non-zero (see
        :func:`~fynance.metrics.ratios._safe_ratio`).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> b_ret = rng.normal(0., 0.01, 999)
    >>> B = 100. * np.cumprod(1. + b_ret)
    >>> B = np.concatenate([[100.], B])
    >>> x_ret = 2. * _compute_returns(B, False)[1:]
    >>> X = 100. * np.cumprod(1. + x_ret)
    >>> X = np.concatenate([[100.], X])
    >>> round(beta(X, B), 4)
    2.0

    See Also
    --------
    alpha, roll_beta_benchmark

    """
    X, B = _validate_curves(X, B)
    x_ret, b_ret = _active_returns(X, B)

    return _ols_beta(x_ret, b_ret)


def alpha(X: NDArray, B: NDArray, period: int = 252, rf: float = 0.0) -> float:
    r""" Annualized Jensen's alpha of the strategy against the benchmark.

    Notes
    -----
    With :math:`x`/:math:`b` the strategy's/benchmark's simple returns, and
    :math:`\beta` the OLS beta (:func:`beta`):

    .. math::

        \alpha = period \times E\left[
            \left(x - \frac{rf}{period}\right)
            - \beta \left(b - \frac{rf}{period}\right)
        \right]

    i.e. the mean per-bar residual of the strategy's excess return once its
    benchmark-driven component (:math:`\beta \times` the benchmark's excess
    return) is removed, annualized by ``period``.

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1]
        Strategy price/level curve.
    B : np.ndarray[float64, ndim=1]
        Benchmark price/level curve, same length as ``X``.
    period : int, optional
        Number of periods per year, default is 252 (trading days).
    rf : float, optional
        Annualized risk-free rate, default is 0.

    Returns
    -------
    float
        Annualized Jensen's alpha.

    Examples
    --------
    A strategy that mirrors the benchmark one-for-one plus a constant bar
    return ``c`` has alpha close to ``c * period`` and beta close to 1:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> b_ret = rng.normal(0., 0.01, 999)
    >>> B = np.concatenate([[100.], 100. * np.cumprod(1. + b_ret)])
    >>> c = 0.0005
    >>> x_ret = b_ret + c
    >>> X = np.concatenate([[100.], 100. * np.cumprod(1. + x_ret)])
    >>> round(beta(X, B), 2)
    1.0
    >>> round(alpha(X, B, period=252) / (c * 252), 2)
    1.0

    See Also
    --------
    beta, information_ratio

    """
    X, B = _validate_curves(X, B)
    b = _ols_beta(*_active_returns(X, B))
    x_ret, b_ret = _active_returns(X, B)
    excess_x = x_ret - rf / period
    excess_b = b_ret - rf / period

    return float(np.mean(excess_x - b * excess_b) * period)


def tracking_error(X: NDArray, B: NDArray, period: int = 252) -> float:
    r""" Annualized standard deviation of the active (strategy - benchmark) return.

    Notes
    -----
    With :math:`x`/:math:`b` the strategy's/benchmark's simple returns:

    .. math::

        TE = \sqrt{period \times Var(x - b)}

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1]
        Strategy price/level curve.
    B : np.ndarray[float64, ndim=1]
        Benchmark price/level curve, same length as ``X``.
    period : int, optional
        Number of periods per year, default is 252 (trading days).

    Returns
    -------
    float
        Annualized tracking error (always non-negative).

    Examples
    --------
    A strategy identical to its benchmark has a zero tracking error:

    >>> import numpy as np
    >>> X = np.array([100., 102., 101., 105., 110.])
    >>> tracking_error(X, X)
    0.0

    See Also
    --------
    information_ratio, beta

    """
    X, B = _validate_curves(X, B)
    x_ret, b_ret = _active_returns(X, B)
    active = x_ret - b_ret

    return float(np.sqrt(period) * np.std(active, ddof=0))


def information_ratio(X: NDArray, B: NDArray, period: int = 252) -> float:
    r""" Annualized mean active return over the tracking error.

    Notes
    -----
    With :math:`x`/:math:`b` the strategy's/benchmark's simple returns and
    :math:`TE` the tracking error (:func:`tracking_error`):

    .. math::

        IR = \frac{period \times E[x - b]}{TE}

    **Zero-tracking-error convention.** Unlike :func:`~fynance.metrics.ratios.
    sharpe`/:func:`~fynance.metrics.ratios.calmar` (which return ``+inf``/
    ``-inf`` on a zero denominator), a zero tracking error means the strategy
    *is* the benchmark bar-for-bar — there is no active bet to size a ratio
    on — so ``IR`` is defined as ``0.0`` regardless of the (necessarily zero)
    numerator, rather than propagating a signed infinity.

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1]
        Strategy price/level curve.
    B : np.ndarray[float64, ndim=1]
        Benchmark price/level curve, same length as ``X``.
    period : int, optional
        Number of periods per year, default is 252 (trading days).

    Returns
    -------
    float
        Annualized information ratio, or ``0.0`` when the tracking error is
        zero.

    Examples
    --------
    A strategy identical to its benchmark has a zero information ratio (no
    active risk to be compensated for):

    >>> import numpy as np
    >>> X = np.array([100., 102., 101., 105., 110.])
    >>> information_ratio(X, X)
    0.0

    See Also
    --------
    tracking_error, beta

    """
    X, B = _validate_curves(X, B)
    x_ret, b_ret = _active_returns(X, B)
    active = x_ret - b_ret
    te = float(np.sqrt(period) * np.std(active, ddof=0))

    if te == 0.0:

        return 0.0

    return float(np.mean(active) * period / te)


def capture_ratio(
    X: NDArray, B: NDArray, side: str = 'up', period: int = 252,
) -> float:
    r""" Up- or down-market capture ratio of the strategy against the benchmark.

    Restricts to the bars where the benchmark moved in the direction given by
    ``side`` (``'up'``: :math:`b_t > 0`; ``'down'``: :math:`b_t < 0`),
    annualizes the strategy's and the benchmark's compounded return over
    that same subset of bars, and returns their ratio.

    Notes
    -----
    Let :math:`\mathcal{T}` the set of bars with :math:`b_t > 0` (``'up'``) or
    :math:`b_t < 0` (``'down'``), and :math:`n = |\mathcal{T}|`:

    .. math::

        annGeo(r, \mathcal{T}) = \left(\prod_{t \in \mathcal{T}}
            (1 + r_t)\right)^{period / n} - 1

        captureRatio = \frac{annGeo(x, \mathcal{T})}{annGeo(b, \mathcal{T})}

    A value above 1 (``'up'``) means the strategy out-gained the benchmark on
    the benchmark's up bars; a value below 1 (``'down'``) means the strategy
    lost less than the benchmark on its down bars — both read as "better" for
    the strategy.

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1]
        Strategy price/level curve.
    B : np.ndarray[float64, ndim=1]
        Benchmark price/level curve, same length as ``X``.
    side : {'up', 'down'}, optional
        Which benchmark bars to condition on. Default is ``'up'``.
    period : int, optional
        Number of periods per year, default is 252 (trading days).

    Returns
    -------
    float
        The capture ratio, ``1.0`` when the strategy replicates the
        benchmark on the selected bars, ``nan`` if the benchmark never moves
        in the requested direction (empty subset).

    Examples
    --------
    A strategy identical to its benchmark has both capture ratios equal to 1:

    >>> import numpy as np
    >>> X = np.array([100., 102., 101., 105., 103., 108.])
    >>> round(capture_ratio(X, X, side='up'), 6)
    1.0
    >>> round(capture_ratio(X, X, side='down'), 6)
    1.0

    See Also
    --------
    benchmark_summary, beta

    """
    if side not in ('up', 'down'):

        raise ValueError(f"side must be 'up' or 'down', got {side!r}")

    X, B = _validate_curves(X, B)
    x_ret, b_ret = _active_returns(X, B)
    mask = b_ret > 0.0 if side == 'up' else b_ret < 0.0
    x_ann = _ann_geo_return(x_ret[mask], period)
    b_ann = _ann_geo_return(b_ret[mask], period)

    return float(_safe_ratio(np.asarray(x_ann), np.asarray(b_ann)))


def benchmark_summary(
    X: NDArray, B: NDArray, period: int = 252, rf: float = 0.0,
) -> dict[str, float]:
    """ Standard benchmark-relative performance summary.

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1]
        Strategy price/level curve.
    B : np.ndarray[float64, ndim=1]
        Benchmark price/level curve, same length as ``X``.
    period : int, optional
        Number of periods per year, default is 252 (trading days).
    rf : float, optional
        Annualized risk-free rate passed to :func:`alpha`. Default is 0.

    Returns
    -------
    dict of str to float
        ``beta``, ``alpha``, ``tracking_error``, ``information_ratio``,
        ``up_capture`` and ``down_capture``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([100., 102., 101., 105., 103., 108.])
    >>> s = benchmark_summary(X, X)
    >>> sorted(s)
    ['alpha', 'beta', 'down_capture', 'information_ratio', 'tracking_error', 'up_capture']
    >>> round(s['beta'], 6)
    1.0

    See Also
    --------
    beta, alpha, tracking_error, information_ratio, capture_ratio

    """
    X, B = _validate_curves(X, B)

    return {
        'beta': beta(X, B, period),
        'alpha': alpha(X, B, period, rf),
        'tracking_error': tracking_error(X, B, period),
        'information_ratio': information_ratio(X, B, period),
        'up_capture': capture_ratio(X, B, 'up', period),
        'down_capture': capture_ratio(X, B, 'down', period),
    }


def roll_beta_benchmark(X: NDArray, B: NDArray, w: int = 252) -> NDArray:
    r""" Trailing rolling beta of the strategy against the benchmark.

    Thin wrapper around :func:`fynance.features.roll_functions.roll_beta`
    applied to the simple returns derived from the ``X``/``B`` price curves
    (see module docstring) — no new kernel, the trailing OLS-slope logic is
    entirely reused from :mod:`fynance.features.roll_functions`.

    Parameters
    ----------
    X : np.ndarray[float64, ndim=1]
        Strategy price/level curve.
    B : np.ndarray[float64, ndim=1]
        Benchmark price/level curve, same length as ``X``.
    w : int, optional
        Size of the trailing window, must be an integer >= 2. Default 252.

    Returns
    -------
    np.ndarray[float64, ndim=1]
        Trailing rolling beta, same length as ``X``/``B``. The first ``w - 1``
        entries are ``np.nan`` (insufficient history).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> b_ret = rng.normal(0., 0.01, 300)
    >>> B = np.concatenate([[100.], 100. * np.cumprod(1. + b_ret)])
    >>> X = np.concatenate([[100.], 100. * np.cumprod(1. + 0.5 * b_ret)])
    >>> rb = roll_beta_benchmark(X, B, w=63)
    >>> bool(np.isnan(rb[:62]).all())
    True
    >>> round(float(np.nanmean(rb[62:])), 1)
    0.5

    See Also
    --------
    beta, fynance.features.roll_functions.roll_beta

    """
    X, B = _validate_curves(X, B)
    x_ret, b_ret = _active_returns(X, B)

    return roll_beta(x_ret, b_ret, w)
