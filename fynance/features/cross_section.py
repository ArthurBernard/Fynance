#!/usr/bin/env python3
# coding: utf-8

""" Cross-sectional transforms on ``(T, N)`` panels.

Per-bar operators for factor research: given a panel of ``T`` bars by ``N``
assets, each transform reduces/normalizes the row (the bar) *across assets*,
independently of every other row. They are the standard building blocks
turning a raw factor into a tradeable cross-sectional signal — rank it,
standardize it, demean it (dollar-neutral), clip its tails, or strip out an
unwanted exposure (e.g. sector or beta).

All transforms are **NaN-aware**: an asset missing at bar ``t`` (``NaN`` in
``X[t]``) is excluded from that bar's statistic and stays ``NaN`` in the
output — it never contaminates, and is never assigned, a value. All
transforms are also **purely cross-sectional**: row ``t`` of the output
depends only on row ``t`` of the input, so they are trivially causal and
row-independent (no lookahead, no leakage across bars).

Main entry points
------------------
- :func:`cs_rank` — per-bar rank (with ``pct=True``, a rank in ``[0, 1]``).
- :func:`cs_zscore` — per-bar standardization.
- :func:`cs_demean` — per-bar (weighted) demeaning.
- :func:`cs_winsorize` — per-bar tail clipping at empirical quantiles.
- :func:`cs_neutralize` — per-bar OLS residual against one or more exposures.

"""

from __future__ import annotations

# Built-in packages
import warnings

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray

# Local packages

__all__ = ['cs_demean', 'cs_neutralize', 'cs_rank', 'cs_winsorize', 'cs_zscore']


# =========================================================================== #
#                                  helpers                                    #
# =========================================================================== #


def _validate_panel(X) -> NDArray[np.float64]:
    """ Cast to float64 and require a 2-D ``(T, N)`` panel.

    Parameters
    ----------
    X : array_like
        Candidate panel.

    Returns
    -------
    np.ndarray
        Validated ``(T, N)`` float64 array.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D.

    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:

        raise ValueError(f"X must be a 2-D (T, N) panel, got shape {X.shape}")

    return X


# =========================================================================== #
#                               Numba kernels                                 #
# =========================================================================== #


@njit(cache=True)
def _rank_row_avg(row: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Average-tie rank of the non-NaN entries of a single row.

    Scipy-free tie-averaged ranking: sort the valid values, then average the
    1-based ranks within each run of equal values. ``NaN`` entries are
    excluded from the ranking and stay ``NaN`` in the output.
    """
    n = row.shape[0]
    out = np.full(n, np.nan)
    idx = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        if not np.isnan(row[i]):
            idx[m] = i
            m += 1

    if m == 0:

        return out

    vals = np.empty(m, dtype=np.float64)
    for k in range(m):
        vals[k] = row[idx[k]]

    order = np.argsort(vals)
    ranks = np.empty(m, dtype=np.float64)
    i = 0
    while i < m:
        j = i
        while j + 1 < m and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    for k in range(m):
        out[idx[k]] = ranks[k]

    return out


@njit(cache=True)
def _cs_rank_kernel(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Row-by-row average-tie rank of a ``(T, N)`` panel (NaN-aware). """
    T, N = X.shape
    out = np.empty((T, N), dtype=np.float64)
    for t in range(T):
        out[t] = _rank_row_avg(X[t])

    return out


# =========================================================================== #
#                            cross-sectional rank                            #
# =========================================================================== #


def cs_rank(X, pct: bool = True) -> NDArray[np.float64]:
    r""" Per-bar cross-sectional rank, NaN-aware.

    At each bar ``t``, the ``n_valid`` non-``NaN`` assets are ranked from
    smallest (rank 1) to largest (rank ``n_valid``); ties get the average of
    the ranks they span (e.g. two tied values at ranks 2 and 3 both get 2.5).
    Missing assets are excluded from the ranking and stay ``NaN``.

    Notes
    -----
    The tie-averaged rank is computed without SciPy: values are sorted with
    :func:`numpy.argsort`, then each run of equal values in sorted order is
    assigned the mean of the 1-based ranks it occupies.

    With ``pct=True`` the rank is rescaled to ``[0, 1]``:

    .. math::

        cs\_rank^{pct}_t(i) = \frac{rank_t(i) - 1}{n\_valid_t - 1}

    so the smallest valid value maps to 0, the largest to 1. When
    ``n_valid_t == 1`` there is no spread to normalize against, so the single
    valid entry maps to 0.5 by convention.

    Parameters
    ----------
    X : array_like
        Panel, shape ``(T, N)``.
    pct : bool, optional
        If ``True`` (default), rescale ranks to ``[0, 1]`` per bar. If
        ``False``, return the raw average rank in ``[1, n_valid]``.

    Returns
    -------
    np.ndarray
        ``(T, N)`` array of ranks, ``NaN`` where ``X`` is ``NaN``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[3., 1., 2.]])
    >>> cs_rank(X, pct=True)
    array([[1. , 0. , 0.5]])
    >>> cs_rank(X, pct=False)
    array([[3., 1., 2.]])

    Ties get the average rank, e.g. the two tied 1's here share ranks 1 and 2:

    >>> cs_rank(np.array([[1., 1., 2.]]), pct=False)
    array([[1.5, 1.5, 3. ]])

    A single valid asset in the bar maps to 0.5 under ``pct=True``:

    >>> cs_rank(np.array([[5., np.nan, np.nan]]))
    array([[0.5, nan, nan]])

    See Also
    --------
    cs_zscore, cs_winsorize

    """
    X = _validate_panel(X)
    ranks = _cs_rank_kernel(X)

    if not pct:

        return ranks

    n_valid = np.sum(~np.isnan(X), axis=1).astype(np.float64)

    with np.errstate(invalid='ignore', divide='ignore'):
        pct_ranks = (ranks - 1.) / (n_valid[:, None] - 1.)

    pct_ranks = np.where(n_valid[:, None] == 1., 0.5, pct_ranks)
    pct_ranks = np.where(np.isnan(ranks), np.nan, pct_ranks)

    return pct_ranks


# =========================================================================== #
#                           cross-sectional zscore                           #
# =========================================================================== #


def cs_zscore(X, ddof: int = 0) -> NDArray[np.float64]:
    r""" Per-bar cross-sectional z-score, NaN-aware.

    At each bar ``t``, standardizes the valid entries against their own bar's
    mean and standard deviation:

    .. math::

        cs\_zscore_t(i) = \frac{X_t(i) - \mu_t}{\sigma_t}

    where :math:`\mu_t` and :math:`\sigma_t` are the mean and standard
    deviation of the non-``NaN`` entries of bar ``t``. A bar with zero
    cross-sectional dispersion (:math:`\sigma_t = 0`, e.g. all valid assets
    tied) maps every valid entry to ``0.0`` rather than dividing by zero.

    Parameters
    ----------
    X : array_like
        Panel, shape ``(T, N)``.
    ddof : int, optional
        Delta degrees of freedom used for :math:`\sigma_t` (``ddof=0`` is the
        population standard deviation, ``ddof=1`` the sample one). Default 0.

    Returns
    -------
    np.ndarray
        ``(T, N)`` array of z-scores, ``NaN`` where ``X`` is ``NaN``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 2., 3.]])
    >>> cs_zscore(X)
    array([[-1.22474487,  0.        ,  1.22474487]])

    A tied (zero-variance) bar maps every valid entry to 0:

    >>> cs_zscore(np.array([[5., 5., 5.]]))
    array([[0., 0., 0.]])

    NaN-aware: the missing asset is excluded from the bar's mean/std and
    stays NaN:

    >>> cs_zscore(np.array([[1., np.nan, 3.]]))
    array([[-1., nan,  1.]])

    See Also
    --------
    cs_rank, cs_demean

    """
    X = _validate_panel(X)
    mask = ~np.isnan(X)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean = np.nanmean(X, axis=1)
        std = np.nanstd(X, axis=1, ddof=ddof)

    with np.errstate(invalid='ignore', divide='ignore'):
        z = (X - mean[:, None]) / std[:, None]

    z = np.where(std[:, None] == 0., 0., z)
    z = np.where(mask, z, np.nan)

    return z


# =========================================================================== #
#                           cross-sectional demean                           #
# =========================================================================== #


def cs_demean(X, weights=None) -> NDArray[np.float64]:
    r""" Per-bar cross-sectional demeaning, NaN-aware.

    Subtracts, from each valid entry of bar ``t``, the (optionally weighted)
    mean of the valid entries of that bar:

    .. math::

        cs\_demean_t(i) = X_t(i) - \bar{X}_t, \qquad
        \bar{X}_t = \frac{\sum_{j \in valid_t} w_t(j) X_t(j)}
                         {\sum_{j \in valid_t} w_t(j)}

    With ``weights=None`` every valid asset gets equal weight (a plain
    cross-sectional mean) — the usual "dollar-neutral" transform. If the
    total weight of the valid assets in a bar is zero (e.g. long/short
    weights that happen to cancel, or all-zero weights), the bar falls back
    to an equal-weighted mean over its valid assets.

    Parameters
    ----------
    X : array_like
        Panel, shape ``(T, N)``.
    weights : array_like, optional
        Weights, either ``(N,)`` (applied identically at every bar) or
        ``(T, N)`` (time-varying). Weights at positions where ``X`` is
        ``NaN`` are ignored. Default ``None`` (equal weight).

    Returns
    -------
    np.ndarray
        ``(T, N)`` demeaned panel, ``NaN`` where ``X`` is ``NaN``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 2., 3.]])
    >>> cs_demean(X)
    array([[-1.,  0.,  1.]])

    Weighted demeaning:

    >>> cs_demean(np.array([[1., 2., 3.]]), weights=np.array([1., 1., 2.]))
    array([[-1.25, -0.25,  0.75]])

    NaN-aware: the missing asset is excluded from the bar's mean and stays
    NaN:

    >>> cs_demean(np.array([[1., np.nan, 3.]]))
    array([[-1., nan,  1.]])

    See Also
    --------
    cs_zscore, cs_neutralize

    """
    X = _validate_panel(X)
    mask = ~np.isnan(X)
    T, N = X.shape

    if weights is None:
        w = mask.astype(np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

        if weights.ndim == 1:
            if weights.shape[0] != N:

                raise ValueError(
                    f"weights of shape (N,) must have N={N}, "
                    f"got shape {weights.shape}"
                )

            w = np.broadcast_to(weights, (T, N)).copy()

        elif weights.shape == (T, N):
            w = weights.copy()

        else:

            raise ValueError(
                f"weights must have shape ({N},) or {(T, N)}, "
                f"got shape {weights.shape}"
            )

        w[~mask] = 0.

    n_valid = mask.sum(axis=1)
    wsum = w.sum(axis=1)
    fallback = (wsum == 0.) & (n_valid > 0)
    if np.any(fallback):
        w[fallback] = mask[fallback].astype(np.float64)
        wsum[fallback] = n_valid[fallback].astype(np.float64)

    X_filled = np.where(mask, X, 0.)
    with np.errstate(invalid='ignore', divide='ignore'):
        wmean = np.where(
            wsum > 0., (w * X_filled).sum(axis=1) / np.where(wsum > 0., wsum, 1.), 0.,
        )

    out = X - wmean[:, None]

    return np.where(mask, out, np.nan)


# =========================================================================== #
#                          cross-sectional winsorize                         #
# =========================================================================== #


def cs_winsorize(X, alpha: float = 0.05) -> NDArray[np.float64]:
    r""" Per-bar cross-sectional winsorization, NaN-aware.

    Clips each bar's valid entries to their own ``[alpha, 1 - alpha]``
    empirical quantiles (linear interpolation, see :func:`numpy.nanquantile`).

    Parameters
    ----------
    X : array_like
        Panel, shape ``(T, N)``.
    alpha : float, optional
        Tail probability clipped on each side, must be in ``[0, 0.5)``.
        Default 0.05 (5% / 95%).

    Returns
    -------
    np.ndarray
        ``(T, N)`` winsorized panel, ``NaN`` where ``X`` is ``NaN``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 2., 3., 4., 100.]])
    >>> cs_winsorize(X, alpha=0.2)
    array([[ 1.8,  2. ,  3. ,  4. , 23.2]])

    NaN-aware: the missing asset is excluded from the bar's quantiles and
    stays NaN:

    >>> cs_winsorize(np.array([[1., np.nan, 3., 100.]]), alpha=0.25)
    array([[ 2. ,  nan,  3. , 51.5]])

    See Also
    --------
    cs_rank, cs_zscore

    """
    X = _validate_panel(X)

    if not (0. <= alpha < 0.5):

        raise ValueError(f"alpha must be in [0, 0.5), got {alpha}")

    mask = ~np.isnan(X)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        lo = np.nanquantile(X, alpha, axis=1)
        hi = np.nanquantile(X, 1. - alpha, axis=1)

    clipped = np.clip(X, lo[:, None], hi[:, None])

    return np.where(mask, clipped, np.nan)


# =========================================================================== #
#                         cross-sectional neutralize                         #
# =========================================================================== #


def cs_neutralize(X, exposures) -> NDArray[np.float64]:
    r""" Per-bar cross-sectional OLS neutralization, NaN-aware.

    At each bar ``t``, regresses the valid entries of ``X`` on the
    corresponding valid rows of ``exposures`` (plus an intercept) by ordinary
    least squares, and returns the residual — the part of ``X`` orthogonal to
    the exposure(s). This is the standard way to strip an unwanted factor
    (e.g. sector, beta, size) out of a raw signal before ranking it.

    Notes
    -----
    For bar ``t`` with valid-asset design matrix
    :math:`D_t = [\mathbb{1}, F_t] \in \mathbb{R}^{n_t \times (K+1)}`
    (an intercept column plus the ``K`` exposure columns) and valid target
    :math:`y_t`, the fit solves the least-squares problem

    .. math::

        \hat\beta_t = \arg\min_\beta \lVert y_t - D_t \beta \rVert_2^2

    via :func:`numpy.linalg.lstsq`, and the output is
    :math:`y_t - D_t \hat\beta_t`. A bar needs at least :math:`K + 1` valid
    assets to identify the :math:`K + 1` parameters (intercept + ``K``
    exposures); bars with fewer valid assets output ``NaN`` for every asset.
    An asset is valid at bar ``t`` only if both ``X[t]`` and every column of
    ``exposures[t]`` are non-``NaN`` there.

    Parameters
    ----------
    X : array_like
        Panel, shape ``(T, N)``.
    exposures : array_like
        Exposure(s) to neutralize against: ``(T, N)`` for a single factor, or
        ``(T, N, K)`` for ``K`` factors.

    Returns
    -------
    np.ndarray
        ``(T, N)`` residual panel. ``NaN`` where ``X`` (or the exposure) is
        ``NaN``, and for every asset of a bar with fewer than ``K + 1`` valid
        assets.

    Examples
    --------
    A single factor that is an exact linear driver of ``X`` neutralizes to
    (numerically) zero:

    >>> import numpy as np
    >>> f = np.array([[1., 2., 3., 4.]])
    >>> X = 2. + 3. * f
    >>> np.round(cs_neutralize(X, f), 8)
    array([[0., 0., 0., 0.]])

    A bar with fewer valid assets than parameters needed (here 1 valid asset,
    2 needed for an intercept + 1 factor) is entirely NaN:

    >>> X = np.array([[1., np.nan, np.nan]])
    >>> f = np.array([[1., 2., 3.]])
    >>> cs_neutralize(X, f)
    array([[nan, nan, nan]])

    See Also
    --------
    cs_demean, cs_zscore

    """
    X = _validate_panel(X)
    exposures = np.asarray(exposures, dtype=np.float64)
    T, N = X.shape

    if exposures.ndim == 2:
        if exposures.shape != (T, N):

            raise ValueError(
                f"exposures of ndim=2 must have shape {(T, N)}, "
                f"got shape {exposures.shape}"
            )

        exposures = exposures[:, :, np.newaxis]

    elif exposures.ndim == 3:
        if exposures.shape[:2] != (T, N):

            raise ValueError(
                f"exposures of ndim=3 must have shape ({T}, {N}, K), "
                f"got shape {exposures.shape}"
            )

    else:

        raise ValueError(
            f"exposures must be 2-D (T, N) or 3-D (T, N, K), "
            f"got ndim={exposures.ndim}"
        )

    K = exposures.shape[2]
    mask = ~np.isnan(X) & ~np.isnan(exposures).any(axis=2)
    out = np.full((T, N), np.nan)

    for t in range(T):
        row_mask = mask[t]
        n_valid = int(row_mask.sum())

        if n_valid < K + 1:

            continue

        y = X[t, row_mask]
        F = exposures[t, row_mask, :]
        design = np.column_stack((np.ones(n_valid), F))
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        out[t, row_mask] = y - design @ coef

    return out


if __name__ == '__main__':

    import doctest

    doctest.testmod()
