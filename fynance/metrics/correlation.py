#!/usr/bin/env python3
# coding: utf-8

""" Predict-vs-realize correlation metrics (Information Coefficient).

The Information Coefficient (IC) is a predict-then-rule guardrail: it scores how
well a *prediction* lines up with the *realized* outcome. Unlike the risk-adjusted
ratios in :mod:`fynance.metrics.ratios`, which evaluate a single equity curve, the
IC takes two aligned series (a prediction and a realization) and returns their
correlation along the time axis.

"""

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

# Local packages

__all__ = ['information_coefficient']


def _pearson_pair(a: NDArray, b: NDArray) -> float:
    """ Pearson correlation of two 1-D vectors, NaN-robust.

    Drops index pairs where either side is NaN, and returns ``np.nan`` when
    fewer than two valid pairs remain or either side has zero variance.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if a.size < 2:

        return np.nan

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))

    if denom == 0.:

        return np.nan

    return float(np.sum(a * b) / denom)


def _spearman_pair(a: NDArray, b: NDArray) -> float:
    """ Spearman (rank) correlation of two 1-D vectors, NaN-robust.

    Spearman is Pearson computed on the ranks of the valid pairs; ties are
    handled by :func:`scipy.stats.rankdata` (average ranks).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if a.size < 2:

        return np.nan

    return _pearson_pair(rankdata(a), rankdata(b))


_HANDLER = {
    'pearson': _pearson_pair,
    'spearman': _spearman_pair,
}


def information_coefficient(
    pred: NDArray,
    real: NDArray,
    *,
    method: str = 'spearman',
    axis: int = 0,
) -> Any:
    r""" Information Coefficient between a prediction and a realized outcome.

    The Information Coefficient (IC) is the correlation between a forecast and
    what actually happened — a predict-then-rule guardrail measuring how much
    signal a prediction carries. With ``method='spearman'`` (default) it is the
    *rank-IC*: a rank correlation that rewards getting the ordering right while
    ignoring the magnitude/shape of the relationship, which is the relevant
    quantity when the prediction is used to rank assets. With
    ``method='pearson'`` it is the ordinary linear correlation.

    The two inputs must be aligned: ``pred[i]`` is the forecast for the outcome
    ``real[i]``.

    Notes
    -----
    For two aligned samples the IC is

    .. math::

        IC = corr(g(pred),\ g(real))

    where :math:`g` is the identity for ``method='pearson'`` and the rank
    transform for ``method='spearman'`` (Spearman = Pearson on ranks). Index
    pairs where either side is NaN are dropped before the computation.

    **Shape contract.** For 1-D inputs ``(T,)`` the IC is a scalar computed over
    the whole sample. For 2-D panel inputs ``(T, N)`` the *default*
    (``axis=0``) is the **cross-sectional IC per bar**: at each time step the
    prediction is correlated against the realization *across the N assets*,
    returning one IC per bar with shape ``(T,)``. This is the ranking
    use-case — at each rebalancing bar, "did the assets I ranked highest
    actually do best?". Pass ``axis=1`` to instead correlate along the time
    axis per asset, returning shape ``(N,)`` (the per-asset time-series IC).

    Parameters
    ----------
    pred : np.ndarray[dtype, ndim=1 or 2]
        Predictions/forecasts (e.g. a model score). Same shape as ``real``.
    real : np.ndarray[dtype, ndim=1 or 2]
        Realized outcomes (e.g. forward returns). Same shape as ``pred``.
    method : {'spearman', 'pearson'}, optional
        ``'spearman'`` (default) for the rank-IC, ``'pearson'`` for the linear
        IC.
    axis : {0, 1}, optional
        Axis indexing the samples to correlate *over*. For 2-D inputs,
        ``axis=0`` correlates across the second dimension per row
        (cross-sectional IC per bar, shape ``(T,)``) and ``axis=1`` correlates
        across the first dimension per column (time-series IC per asset, shape
        ``(N,)``). Ignored for 1-D inputs. Default is 0.

    Returns
    -------
    float or np.ndarray[np.float64, ndim=1]
        The IC. A scalar for 1-D inputs, a 1-D array for 2-D inputs. Entries
        are ``np.nan`` (never an exception) where fewer than two valid pairs
        remain or either side has zero variance.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Information_coefficient

    Examples
    --------
    A perfect monotonic prediction has a rank-IC of 1, even when the
    relationship is non-linear (only the ordering matters):

    >>> import numpy as np
    >>> real = np.array([1., 2., 3., 4., 5.])
    >>> pred = real ** 3
    >>> float(information_coefficient(pred, real))
    1.0
    >>> round(float(information_coefficient(pred, real, method='pearson')), 4)
    0.9431

    A panel returns one cross-sectional IC per bar — here the ranking is
    correct on the first bar and inverted on the second:

    >>> pred = np.array([[1., 2., 3.], [1., 2., 3.]])
    >>> real = np.array([[1., 2., 3.], [3., 2., 1.]])
    >>> information_coefficient(pred, real)
    array([ 1., -1.])

    See Also
    --------
    sharpe, sortino

    """
    if method not in _HANDLER:

        raise ValueError(
            f"unknown method {method!r}, only 'spearman' and 'pearson' "
            "are supported"
        )

    corr = _HANDLER[method]
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)

    if pred.shape != real.shape:

        raise ValueError(
            f"pred and real must have the same shape, got {pred.shape} and "
            f"{real.shape}"
        )

    if pred.ndim == 1:

        return corr(pred, real)

    if pred.ndim != 2:

        raise ValueError(
            f"only 1-D and 2-D arrays are supported, got ndim={pred.ndim}"
        )

    if axis == 0:
        # Cross-sectional IC per bar: correlate across the N assets within each
        # time step, one value per row.
        return np.array(
            [corr(pred[t], real[t]) for t in range(pred.shape[0])],
            dtype=np.float64,
        )

    elif axis == 1:
        # Time-series IC per asset: correlate along time within each column.
        return np.array(
            [corr(pred[:, n], real[:, n]) for n in range(pred.shape[1])],
            dtype=np.float64,
        )

    raise np.exceptions.AxisError(axis, pred.ndim)
