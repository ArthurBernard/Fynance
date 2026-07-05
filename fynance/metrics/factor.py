#!/usr/bin/env python3
# coding: utf-8

""" Alphalens-style factor evaluation on data-agnostic ``(T, N)`` panels.

A *factor* is a per-bar cross-sectional score (one value per asset per bar) used
to rank assets; these helpers score how much forward-return signal that ranking
carries. They operate on raw NumPy panels — no dates, no asset names — so they
apply to any ``(T, N)`` factor/return pair.

**Alignment convention** (shared with
:func:`fynance.metrics.information_coefficient`): ``factor[t]`` is aligned with
``fwd[t]``, the return realized *after* the factor is known. The caller builds
``fwd`` (e.g. via :func:`fynance.features.horizon_returns`) so that no future
information leaks into the factor.

The suite mirrors a factor tear-sheet: quantile-portfolio returns
(:func:`quantile_returns`), a trailing Information Coefficient
(:func:`roll_information_coefficient`), IC decay across horizons
(:func:`ic_decay`), IC summary statistics (:func:`ic_summary`) and factor rank
autocorrelation (:func:`factor_rank_autocorr`, a turnover proxy).

"""

from __future__ import annotations

# Built-in packages
from dataclasses import dataclass

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.features.horizon import horizon_returns
from fynance.metrics.correlation import information_coefficient

__all__ = [
    'QuantileResult',
    'quantile_returns',
    'roll_information_coefficient',
    'ic_decay',
    'ic_summary',
    'factor_rank_autocorr',
]


@dataclass
class QuantileResult:
    """ Per-bar quantile-portfolio returns of a factor.

    Returned by :func:`quantile_returns`. On each bar the valid assets are
    bucketed into ``n_quantiles`` equal-count quantiles by factor value (bucket
    ``0`` is the lowest factor value, bucket ``n_quantiles - 1`` the highest) and
    the equal-weighted mean forward return of each bucket is recorded.

    Attributes
    ----------
    quantile_returns : np.ndarray[np.float64, ndim=2]
        Shape ``(T, Q)``. Mean forward return per per-bar factor quantile. Bars
        with fewer than ``Q`` valid assets are a full row of ``np.nan``.
    spread : np.ndarray[np.float64, ndim=1]
        Shape ``(T,)``. Top-minus-bottom quantile return
        (``quantile_returns[:, -1] - quantile_returns[:, 0]``), the long-short
        return of the factor.
    counts : np.ndarray[np.int64, ndim=2]
        Shape ``(T, Q)``. Number of assets in each bucket (``0`` on invalid
        bars).
    n_quantiles : int
        Number of quantiles ``Q``.

    """

    quantile_returns: NDArray
    spread: NDArray
    counts: NDArray
    n_quantiles: int


def quantile_returns(
    factor: NDArray,
    fwd: NDArray,
    n_quantiles: int = 5,
) -> QuantileResult:
    r""" Equal-count quantile-portfolio returns of a factor.

    On each bar the assets valid in *both* ``factor`` and ``fwd`` (finite on both
    sides) are sorted by factor value and split into ``n_quantiles`` equal-count
    buckets — ties are broken by rank order (a stable sort), bucket ``0`` holding
    the lowest factor values and bucket ``n_quantiles - 1`` the highest. Each
    bucket's return is the equal-weighted mean of its assets' forward returns.
    Bars with fewer than ``n_quantiles`` valid assets yield a row of ``np.nan``
    (and zero counts).

    **Alignment.** ``factor[t]`` is the score known at bar ``t`` and ``fwd[t]``
    the return realized *after* ``t`` (built by the caller, e.g. via
    :func:`fynance.features.horizon_returns`), so a positive top-minus-bottom
    :attr:`~QuantileResult.spread` means high-factor assets out-earned
    low-factor assets.

    Parameters
    ----------
    factor : np.ndarray[dtype, ndim=2]
        Factor panel ``(T, N)`` — the per-bar cross-sectional score.
    fwd : np.ndarray[dtype, ndim=2]
        Forward-return panel ``(T, N)`` aligned with ``factor``.
    n_quantiles : int, optional
        Number of equal-count buckets ``Q`` (default ``5``), a positive integer
        of at least ``2``.

    Returns
    -------
    QuantileResult
        The per-bar bucket returns, long-short spread, bucket counts and
        ``n_quantiles``.

    Examples
    --------
    A two-bar, four-asset panel split into two buckets; the factor ranking is
    ascending on the first bar and descending on the second:

    >>> import numpy as np
    >>> factor = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    >>> fwd = np.array([[1., 2., 3., 4.], [1., 2., 3., 4.]])
    >>> res = quantile_returns(factor, fwd, n_quantiles=2)
    >>> res.quantile_returns
    array([[1.5, 3.5],
           [3.5, 1.5]])
    >>> res.spread
    array([ 2., -2.])
    >>> res.counts
    array([[2, 2],
           [2, 2]])

    See Also
    --------
    fynance.metrics.information_coefficient, ic_summary

    """
    if not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 2:

        raise ValueError(
            f"n_quantiles must be an integer >= 2, got {n_quantiles!r}"
        )

    factor = np.asarray(factor, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)

    if factor.shape != fwd.shape:

        raise ValueError(
            f"factor and fwd must have the same shape, got {factor.shape} and "
            f"{fwd.shape}"
        )

    if factor.ndim != 2:

        raise ValueError(
            f"factor and fwd must be 2-D (T, N) panels, got ndim={factor.ndim}"
        )

    Q = int(n_quantiles)
    T = factor.shape[0]
    qret = np.full((T, Q), np.nan, dtype=np.float64)
    counts = np.zeros((T, Q), dtype=np.int64)
    spread = np.full(T, np.nan, dtype=np.float64)

    for t in range(T):
        valid = np.isfinite(factor[t]) & np.isfinite(fwd[t])
        n_valid = int(valid.sum())

        if n_valid < Q:
            # Too few assets to fill every bucket -> leave the NaN row.
            continue

        f_valid = factor[t, valid]
        r_valid = fwd[t, valid]

        # Ordinal ranks 0..n_valid-1 (ties broken by original order via a stable
        # sort), then equal-count buckets rank * Q // n_valid in [0, Q-1].
        order = np.argsort(f_valid, kind='stable')
        ranks = np.empty(n_valid, dtype=np.int64)
        ranks[order] = np.arange(n_valid)
        buckets = ranks * Q // n_valid

        for q in range(Q):
            in_q = buckets == q
            counts[t, q] = int(in_q.sum())
            qret[t, q] = float(r_valid[in_q].mean())

        spread[t] = qret[t, Q - 1] - qret[t, 0]

    return QuantileResult(
        quantile_returns=qret,
        spread=spread,
        counts=counts,
        n_quantiles=Q,
    )


def roll_information_coefficient(
    pred: NDArray,
    real: NDArray,
    w: int = 63,
    method: str = 'spearman',
) -> NDArray:
    r""" Trailing-window Information Coefficient.

    A rolling view of the :func:`fynance.metrics.information_coefficient`: rather
    than one number over the whole sample, it reports how the predictive power
    evolves through time over a trailing window of ``w`` bars. It is strictly
    causal — the value at ``t`` uses only data on ``[t - w + 1, t]`` — so the
    first ``w - 1`` entries are ``np.nan``.

    Two input shapes are supported:

    - **1-D** ``(T,)`` — a single aligned ``(pred, real)`` time series; the value
      at ``t`` is the IC computed over the trailing window ``[t - w + 1, t]``.
    - **2-D** ``(T, N)`` panel — the per-bar *cross-sectional* IC is computed
      first (across the ``N`` assets, one value per bar, reusing
      :func:`information_coefficient`), then smoothed by a trailing mean over
      ``w`` bars (NaN bars are ignored within the window).

    **Alignment.** ``pred[t]`` is the score known at ``t`` and ``real[t]`` the
    outcome realized after ``t``.

    Parameters
    ----------
    pred : np.ndarray[dtype, ndim=1 or 2]
        Predictions/scores. Same shape as ``real``.
    real : np.ndarray[dtype, ndim=1 or 2]
        Realized outcomes (e.g. forward returns).
    w : int, optional
        Trailing window length in bars (default ``63``), a positive integer.
    method : {'spearman', 'pearson'}, optional
        Correlation used for the IC — ``'spearman'`` (default) for the rank-IC,
        ``'pearson'`` for the linear IC.

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Trailing IC of shape ``(T,)``; the first ``w - 1`` entries are
        ``np.nan``.

    Examples
    --------
    On a monotone 1-D series every trailing window is perfectly ordered, so the
    rank-IC is ``1`` once the window is full:

    >>> import numpy as np
    >>> pred = np.arange(10.)
    >>> real = np.arange(10.)
    >>> ic = roll_information_coefficient(pred, real, w=5)
    >>> bool(np.isnan(ic[:4]).all())
    True
    >>> bool(np.allclose(ic[4:], 1.0))
    True

    See Also
    --------
    fynance.metrics.information_coefficient, ic_summary

    """
    if not isinstance(w, (int, np.integer)) or w < 1:

        raise ValueError(f"w must be a positive integer, got {w!r}")

    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)

    if pred.shape != real.shape:

        raise ValueError(
            f"pred and real must have the same shape, got {pred.shape} and "
            f"{real.shape}"
        )

    if pred.ndim not in (1, 2):

        raise ValueError(
            f"only 1-D and 2-D arrays are supported, got ndim={pred.ndim}"
        )

    w = int(w)
    T = pred.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)

    if pred.ndim == 1:
        for t in range(w - 1, T):
            out[t] = information_coefficient(
                pred[t - w + 1:t + 1], real[t - w + 1:t + 1], method=method,
            )

        return out

    # 2-D: per-bar cross-sectional IC, then trailing mean over w (NaN-robust).
    ic_bar = np.asarray(
        information_coefficient(pred, real, method=method), dtype=np.float64,
    )

    for t in range(w - 1, T):
        window = ic_bar[t - w + 1:t + 1]

        if np.any(np.isfinite(window)):
            out[t] = float(np.nanmean(window))

    return out


def ic_decay(
    factor: NDArray,
    prices: NDArray,
    horizons: tuple[int, ...] = (1, 5, 10, 21),
    method: str = 'spearman',
) -> NDArray:
    r""" Information Coefficient decay across forward horizons.

    Measures how quickly a factor's predictive power fades as the prediction
    horizon lengthens. For each horizon ``h`` the mean per-bar cross-sectional IC
    is computed between the factor and the ``h``-bar **non-overlapping** forward
    return from :func:`fynance.features.horizon_returns` (non-overlapping so the
    labels do not share price moves and inflate the IC). A signal with real
    short-horizon edge shows a high IC at ``h = 1`` that decays toward zero as
    ``h`` grows.

    **Alignment.** ``factor[t]`` is aligned with the forward return starting at
    ``t``; both share the same time index (the first axis of ``prices``).

    Parameters
    ----------
    factor : np.ndarray[dtype, ndim=2]
        Factor panel ``(T, N)``.
    prices : np.ndarray[dtype, ndim=2]
        Price panel ``(T, N)`` from which the forward returns are built; same
        time index as ``factor``.
    horizons : tuple of int, optional
        Forward horizons in bars (default ``(1, 5, 10, 21)``). A horizon that is
        not shorter than ``T`` yields ``np.nan``.
    method : {'spearman', 'pearson'}, optional
        Correlation used for the IC (default ``'spearman'``).

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Mean IC per horizon, shape ``(len(horizons),)``.

    Examples
    --------
    A factor equal to the realized one-bar forward return is a perfect
    one-bar predictor, so its IC at horizon 1 is ``1``:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> prices = 100. * np.cumprod(1. + rng.normal(0., 0.01, (200, 5)), axis=0)
    >>> fwd1 = prices[1:] / prices[:-1] - 1.
    >>> factor = np.vstack([fwd1, np.full((1, 5), np.nan)])
    >>> decay = ic_decay(factor, prices, horizons=(1, 5))
    >>> bool(decay[0] > 0.99)
    True

    See Also
    --------
    fynance.features.horizon_returns, fynance.metrics.information_coefficient

    """
    factor = np.asarray(factor, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)

    if factor.shape != prices.shape:

        raise ValueError(
            f"factor and prices must have the same shape, got {factor.shape} "
            f"and {prices.shape}"
        )

    if factor.ndim != 2:

        raise ValueError(
            f"factor and prices must be 2-D (T, N) panels, got "
            f"ndim={factor.ndim}"
        )

    T = factor.shape[0]
    decay = np.full(len(horizons), np.nan, dtype=np.float64)

    for i, h in enumerate(horizons):
        if T <= h:
            # horizon_returns needs strictly more than h bars; leave NaN.
            continue

        fwd = horizon_returns(prices, h)
        # Non-overlapping labels keep one sample every h bars starting at 0; the
        # factor is sampled on the very same base indices to stay aligned.
        base_idx = np.arange(0, T - h, h)
        ic_bar = np.asarray(
            information_coefficient(factor[base_idx], fwd, method=method),
            dtype=np.float64,
        )

        if np.any(np.isfinite(ic_bar)):
            decay[i] = float(np.nanmean(ic_bar))

    return decay


def ic_summary(
    pred: NDArray,
    real: NDArray,
    method: str = 'spearman',
) -> dict[str, float]:
    r""" Summary statistics of the per-bar cross-sectional Information Coefficient.

    Reduces a ``(T, N)`` panel to the headline numbers of a factor tear-sheet.
    The per-bar cross-sectional IC (one value per bar, across the ``N`` assets)
    is computed with :func:`fynance.metrics.information_coefficient`, then
    summarized. ``icir`` (the IC information ratio) and ``t_stat`` gauge whether
    the mean IC is distinguishable from zero given its bar-to-bar variability.

    **Alignment.** ``pred[t]`` is the score known at ``t`` and ``real[t]`` the
    outcome realized after ``t``.

    Parameters
    ----------
    pred : np.ndarray[dtype, ndim=2]
        Factor/score panel ``(T, N)``.
    real : np.ndarray[dtype, ndim=2]
        Aligned forward-return panel ``(T, N)``.
    method : {'spearman', 'pearson'}, optional
        Correlation used for the IC (default ``'spearman'``).

    Returns
    -------
    dict of str to float
        ``mean_ic`` (mean per-bar IC), ``icir`` (mean / std of the per-bar IC),
        ``t_stat`` (``icir * sqrt(n_bars)``), ``hit_rate`` (share of bars with a
        positive IC) and ``n_bars`` (number of bars with a finite IC). ``icir``
        and ``t_stat`` are ``nan`` when the IC has zero variance.

    Notes
    -----
    With :math:`IC_t` the per-bar IC over the :math:`n` bars with a finite value,

    .. math::

        \mathrm{ICIR} = \frac{\overline{IC}}{\sigma_{IC}}, \qquad
        t = \mathrm{ICIR}\,\sqrt{n}

    where :math:`\sigma_{IC}` is the sample standard deviation (``ddof=1``).

    Examples
    --------
    A factor equal to the realized outcome has a perfect per-bar IC:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> real = rng.normal(size=(100, 10))
    >>> pred = real.copy()
    >>> s = ic_summary(pred, real)
    >>> round(s['mean_ic'], 6)
    1.0
    >>> round(s['hit_rate'], 2)
    1.0
    >>> s['n_bars']
    100

    See Also
    --------
    fynance.metrics.information_coefficient, roll_information_coefficient

    """
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)

    if pred.shape != real.shape:

        raise ValueError(
            f"pred and real must have the same shape, got {pred.shape} and "
            f"{real.shape}"
        )

    if pred.ndim != 2:

        raise ValueError(
            f"pred and real must be 2-D (T, N) panels, got ndim={pred.ndim}"
        )

    ic_bar = np.asarray(
        information_coefficient(pred, real, method=method), dtype=np.float64,
    )
    finite = ic_bar[np.isfinite(ic_bar)]
    n_bars = int(finite.size)

    if n_bars == 0:

        return {
            'mean_ic': float('nan'),
            'icir': float('nan'),
            't_stat': float('nan'),
            'hit_rate': float('nan'),
            'n_bars': 0,
        }

    mean_ic = float(finite.mean())
    std_ic = float(finite.std(ddof=1)) if n_bars > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0.0 else float('nan')
    t_stat = icir * np.sqrt(n_bars) if std_ic > 0.0 else float('nan')
    hit_rate = float(np.mean(finite > 0.0))

    return {
        'mean_ic': mean_ic,
        'icir': float(icir),
        't_stat': float(t_stat),
        'hit_rate': hit_rate,
        'n_bars': n_bars,
    }


def factor_rank_autocorr(factor: NDArray, lag: int = 1) -> NDArray:
    r""" Cross-sectional rank autocorrelation of a factor (turnover proxy).

    At each bar the Spearman (rank) correlation between the factor's
    cross-section at ``t`` and at ``t - lag`` measures how much the ranking is
    preserved from one bar to the next. A value near ``1`` means a stable
    ranking (low turnover); a value near ``0`` means the ranking is reshuffled
    each bar (high turnover). It is the standard turnover proxy of a factor
    tear-sheet.

    Parameters
    ----------
    factor : np.ndarray[dtype, ndim=2]
        Factor panel ``(T, N)``.
    lag : int, optional
        Lag in bars between the two cross-sections (default ``1``), a positive
        integer.

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Rank autocorrelation per bar, shape ``(T,)``. The first ``lag`` entries
        are ``np.nan``, as is any bar where either cross-section has fewer than
        three finite entries.

    Examples
    --------
    A factor whose ranking never changes has a rank autocorrelation of ``1``:

    >>> import numpy as np
    >>> factor = np.tile(np.arange(5.), (4, 1))
    >>> ac = factor_rank_autocorr(factor, lag=1)
    >>> float(ac[0])
    nan
    >>> bool(np.allclose(ac[1:], 1.0))
    True

    See Also
    --------
    fynance.metrics.information_coefficient

    """
    if not isinstance(lag, (int, np.integer)) or lag < 1:

        raise ValueError(f"lag must be a positive integer, got {lag!r}")

    factor = np.asarray(factor, dtype=np.float64)

    if factor.ndim != 2:

        raise ValueError(
            f"factor must be a 2-D (T, N) panel, got ndim={factor.ndim}"
        )

    lag = int(lag)
    T = factor.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)

    for t in range(lag, T):
        now = factor[t]
        past = factor[t - lag]

        if np.isfinite(now).sum() < 3 or np.isfinite(past).sum() < 3:
            # Too few ranks on either side for a meaningful rank correlation.
            continue

        out[t] = information_coefficient(now, past, method='spearman')

    return out
