#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Block / stationary bootstrap for dependent (autocorrelated) return series.

:func:`~fynance.research.guards.permutation_test` builds its null by shuffling
returns **i.i.d.**, which destroys *all* temporal structure -- a fine null for
"is this edge just from the marginal distribution of returns", but too harsh a
null when the returns themselves are genuinely autocorrelated (e.g. slow
regimes, volatility clustering): an i.i.d. shuffle would flag that
autocorrelation itself as an artifact.

This module resamples **blocks** of returns instead of individual observations,
so short-range dependence survives the resampling:

- :func:`resample_paths` -- the resampling primitive (fixed-length ``circular``
  blocks, or ``stationary`` blocks with a random, geometrically-distributed
  length -- Politis & Romano, *The Stationary Bootstrap*, JASA 1994).
- :func:`bootstrap_metric` -- a percentile confidence interval for any metric,
  built from :func:`resample_paths`.
- :func:`block_permutation_test` -- the dependence-preserving analogue of
  :func:`~fynance.research.guards.permutation_test`: is the observed mean return
  larger than a block-bootstrap null centered at zero, i.e. is there a genuine
  edge once autocorrelation is accounted for.

All functions are data-agnostic and Numba-accelerated; nothing here reads real
data or stores results.

"""

# Built-in
from __future__ import annotations

from typing import Any, Callable

# Third-party
import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ['resample_paths', 'bootstrap_metric', 'block_permutation_test']


def _to_array(data: Any) -> NDArray[np.float64]:
    """ Coerce an array-like of returns to a 1-D float64 array. """
    return np.asarray(data, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------- #
#   numba kernels                                                              #
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _circular_kernel(returns, starts, block, T):
    """ Concatenate fixed-length, wrap-around blocks starting at ``starts``. """
    n_paths, n_blocks = starts.shape
    out = np.empty((n_paths, T), dtype=np.float64)
    for p in range(n_paths):
        pos = 0
        for b in range(n_blocks):
            s = starts[p, b]
            for j in range(block):
                if pos >= T:
                    break
                out[p, pos] = returns[(s + j) % T]
                pos += 1
    return out


@njit(cache=True)
def _stationary_kernel(returns, starts, u, p, T):
    """ Politis & Romano (1994) stationary bootstrap.

    At each step, with probability ``p`` a fresh (uniformly drawn) block start
    is used; otherwise the previous index is continued (wrapping circularly),
    so realized block lengths are geometrically distributed with mean ``1/p``.

    """
    n_paths = starts.shape[0]
    out = np.empty((n_paths, T), dtype=np.float64)
    for k in range(n_paths):
        idx = starts[k, 0]
        out[k, 0] = returns[idx]
        for t in range(1, T):
            if u[k, t] < p:
                idx = starts[k, t]
            else:
                idx = (idx + 1) % T
            out[k, t] = returns[idx]
    return out


# --------------------------------------------------------------------------- #
#   public API                                                                 #
# --------------------------------------------------------------------------- #


def resample_paths(
    returns: Any,
    *,
    n_paths: int = 1000,
    block: int = 21,
    method: str = "stationary",
    seed: int = 0,
) -> NDArray[np.float64]:
    """ Block-bootstrap resampled paths of a (possibly autocorrelated) series.

    Unlike an i.i.d. shuffle, resampling whole blocks preserves the short-range
    dependence *within* a block (e.g. volatility clustering, slow trends), which
    makes the resampled paths a more honest stand-in for "another draw from the
    same data-generating process" than a fully shuffled series.

    Two block schemes are supported:

    - ``'circular'``: fixed-length blocks of ``block`` observations, each
      starting at a uniformly-drawn index and wrapping circularly around the
      series (Politis & Romano, 1992). ``ceil(T / block)`` blocks are drawn and
      concatenated, then truncated to length ``T``.
    - ``'stationary'``: blocks of *random*, geometrically-distributed length
      with mean ``block`` (Politis & Romano, 1994): at each step, with
      probability ``1 / block`` a new block starts at a fresh uniformly-drawn
      index, otherwise the current block continues (wrapping circularly). This
      scheme yields a strictly stationary resampled process, unlike the
      fixed-length circular scheme (which has a seam at every block boundary).

    All randomness (block starts, and for ``'stationary'`` the per-step
    continue/restart draws) is generated once via
    ``numpy.random.default_rng(seed)`` *before* entering the Numba kernel, so
    the result is fully reproducible for a fixed ``seed``.

    Parameters
    ----------
    returns : array-like
        Return series to resample, shape ``(T,)``.
    n_paths : int
        Number of resampled paths to draw.
    block : int
        Block length (``'circular'``) or mean block length (``'stationary'``).
    method : {'circular', 'stationary'}
        Resampling scheme, see above.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    numpy.ndarray, shape (n_paths, T)
        Resampled paths (each row is one bootstrap replicate of ``returns``).

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 2 observations, ``n_paths`` or ``block``
        is not a positive integer, or ``method`` is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.research import resample_paths
    >>> r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> paths = resample_paths(r, n_paths=3, block=2, method='circular', seed=0)
    >>> paths.shape
    (3, 5)
    >>> a = resample_paths(r, n_paths=2, seed=1)
    >>> b = resample_paths(r, n_paths=2, seed=1)
    >>> bool(np.array_equal(a, b))
    True

    """
    r = _to_array(returns)
    T = r.size
    if T < 2:
        raise ValueError(f"returns must have at least 2 observations, got {T}")

    if n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths}")

    if block < 1:
        raise ValueError(f"block must be a positive integer, got {block}")

    rng = np.random.default_rng(seed)

    if method == "circular":
        n_blocks = -(-T // block)  # ceil division
        starts = rng.integers(0, T, size=(n_paths, n_blocks))

        return _circular_kernel(r, starts, block, T)

    if method == "stationary":
        starts = rng.integers(0, T, size=(n_paths, T))
        u = rng.random((n_paths, T))
        p = 1.0 / block

        return _stationary_kernel(r, starts, u, p, T)

    raise ValueError(f"method must be 'circular' or 'stationary', got {method!r}")


def bootstrap_metric(
    returns: Any,
    metric: Callable[[NDArray[np.float64]], float],
    *,
    n_paths: int = 1000,
    block: int = 21,
    method: str = "stationary",
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """ Block-bootstrap percentile confidence interval for an arbitrary metric.

    Computes ``metric`` on the observed ``returns``, then again on
    ``n_paths`` block-bootstrap replicates from :func:`resample_paths`, and
    reports the percentile confidence interval of the replicate distribution.

    Parameters
    ----------
    returns : array-like
        Return series, shape ``(T,)``.
    metric : callable
        ``returns -> float``, e.g. ``numpy.mean`` or a custom Sharpe-like
        statistic. Applied identically to the observed series and to every
        resampled path.
    n_paths : int
        Number of bootstrap replicates.
    block : int
        Block length (``'circular'``) or mean block length (``'stationary'``),
        forwarded to :func:`resample_paths`.
    method : {'circular', 'stationary'}
        Resampling scheme, forwarded to :func:`resample_paths`.
    ci : float
        Confidence level in ``(0, 1)``, e.g. ``0.95`` for a 95% CI.
    seed : int
        Seed for reproducibility, forwarded to :func:`resample_paths`.

    Returns
    -------
    dict
        ``estimate`` (metric on the observed data), ``lo``/``hi`` (percentile
        CI bounds), ``distribution`` (the ``(n_paths,)`` array of per-replicate
        metric values).

    Raises
    ------
    ValueError
        If ``ci`` is not in ``(0, 1)`` (other invalid inputs are caught by
        :func:`resample_paths`).

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.research import bootstrap_metric
    >>> r = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0, 0.01])
    >>> out = bootstrap_metric(r, np.mean, n_paths=500, block=2, seed=0)
    >>> sorted(out)
    ['distribution', 'estimate', 'hi', 'lo']
    >>> round(out['estimate'], 5)
    0.00375
    >>> out['distribution'].shape
    (500,)
    >>> out['lo'] <= out['hi']
    True

    """
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    r = _to_array(returns)
    estimate = float(metric(r))

    paths = resample_paths(r, n_paths=n_paths, block=block, method=method, seed=seed)
    distribution = np.array(
        [metric(paths[i]) for i in range(paths.shape[0])], dtype=np.float64
    )

    alpha = 1.0 - ci
    lo, hi = np.quantile(distribution, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "estimate": estimate,
        "lo": float(lo),
        "hi": float(hi),
        "distribution": distribution,
    }


def block_permutation_test(
    strategy_returns: Any,
    *,
    n_perm: int = 1000,
    block: int = 21,
    method: str = "stationary",
    seed: int = 0,
) -> float:
    """ Dependence-preserving analogue of :func:`~fynance.research.guards.permutation_test`.

    ``guards.permutation_test`` reruns a :class:`~fynance.strategy.Strategy` on
    i.i.d.-shuffled price paths -- an appropriate null when the question is
    "does this strategy exploit *any* real structure", but it also destroys
    genuine autocorrelation, which would bias the null toward rejecting a real,
    dependence-driven edge as noise.

    ``block_permutation_test`` instead takes an already-computed strategy
    **return series** (no strategy rerun) and tests whether its mean is
    significantly positive against a null that *preserves* autocorrelation:
    the observed series is demeaned, then block-bootstrap resampled via
    :func:`resample_paths` (so within-block dependence survives), giving the
    distribution of the mean statistic expected **under no drift**. The
    p-value uses the same smoothed convention as
    :func:`~fynance.research.guards.permutation_test`:
    ``(#{null >= observed} + 1) / (n_perm + 1)``.

    Parameters
    ----------
    strategy_returns : array-like
        Realized return series of the strategy under test, shape ``(T,)``.
    n_perm : int
        Number of block-bootstrap replicates forming the null distribution.
    block : int
        Block length (``'circular'``) or mean block length (``'stationary'``),
        forwarded to :func:`resample_paths`.
    method : {'circular', 'stationary'}
        Resampling scheme, forwarded to :func:`resample_paths`.
    seed : int
        Master seed, forwarded to :func:`resample_paths` (reproducible for a
        fixed seed).

    Returns
    -------
    float
        Smoothed p-value for the one-sided test "the observed mean return is
        larger than chance would produce, given the series' own autocorrelation
        structure". Low values indicate a genuine (not merely
        autocorrelation-driven) positive edge.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.research import block_permutation_test
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.standard_normal(300) * 0.01
    >>> p = block_permutation_test(noise, n_perm=200, block=10, seed=1)
    >>> 0.0 < p <= 1.0
    True

    """
    r = _to_array(strategy_returns)
    observed = float(np.mean(r))
    centered = r - observed

    null_paths = resample_paths(
        centered, n_paths=n_perm, block=block, method=method, seed=seed
    )
    null = null_paths.mean(axis=1)

    return float((np.sum(null >= observed) + 1) / (n_perm + 1))
