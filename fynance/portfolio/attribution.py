#!/usr/bin/env python3
# coding: utf-8

r""" Ex-ante risk decomposition for portfolios.

Marginal and absolute risk contributions of a weight vector to portfolio
variance, plus a causal rolling variant for walk-forward decomposition of
a `(T, N)` weight path against sliding windows of returns.

Main entry points
-----------------
- :func:`marginal_risk` — marginal contribution of each asset to portfolio
  volatility.
- :func:`risk_contribution` — absolute (or percentage) risk contribution
  of each asset.
- :func:`roll_risk_contribution` — causal rolling risk contributions over
  a time series.

"""

from __future__ import annotations

# Built-in packages
from typing import Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = ['marginal_risk', 'risk_contribution', 'roll_risk_contribution']


def marginal_risk(w: NDArray, sigma: NDArray) -> NDArray:
    r""" Marginal contribution of each asset to portfolio volatility.

    Compute the gradient of portfolio volatility with respect to weights:
    :math:`\partial \sigma_p / \partial w = (\Sigma w) / \sigma_p`, where
    :math:`\sigma_p = \sqrt{w^\top \Sigma w}` is the portfolio volatility.

    Parameters
    ----------
    w : array_like
        Portfolio weights, shape ``(N,)`` or ``(N, 1)``.
    sigma : array_like
        Symmetric ``(N, N)`` covariance matrix.

    Returns
    -------
    np.ndarray
        Marginal risk per asset, shape ``(N,)``. If portfolio volatility
        is zero, returns zeros.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([0.6, 0.4])
    >>> sigma = np.array([[0.04, 0.0], [0.0, 0.01]])
    >>> mr = marginal_risk(w, sigma)
    >>> mr.shape
    (2,)
    >>> bool(np.all(mr > 0))
    True

    """
    w = np.asarray(w, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square 2-D covariance matrix.")
    if sigma.shape[0] != w.shape[0]:
        raise ValueError(
            f"sigma shape {sigma.shape} does not match w shape {w.shape}."
        )
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma contains non-finite values (NaN or inf).")
    if not np.all(np.isfinite(w)):
        raise ValueError("w contains non-finite values (NaN or inf).")

    sigma_w = sigma @ w
    sigma_p = np.sqrt(w @ sigma_w)

    if sigma_p == 0.0:
        return np.zeros_like(w)

    return sigma_w / sigma_p


def risk_contribution(
    w: NDArray,
    sigma: NDArray,
    pct: bool = True,
) -> NDArray:
    r""" Risk contribution of each asset to portfolio variance.

    Absolute risk contribution: :math:`RC_i = w_i \cdot MR_i`, where
    :math:`MR_i` is the marginal risk. When :math:`pct=True`, normalize
    by portfolio volatility so contributions sum to 1 (percentage).

    Parameters
    ----------
    w : array_like
        Portfolio weights, shape ``(N,)`` or ``(N, 1)``.
    sigma : array_like
        Symmetric ``(N, N)`` covariance matrix.
    pct : bool, optional
        If True (default), return percentage contributions that sum to 1.
        If False, return absolute contributions that sum to portfolio
        volatility.

    Returns
    -------
    np.ndarray
        Risk contribution per asset, shape ``(N,)``. If portfolio
        volatility is zero, returns zeros.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([0.5, 0.5])
    >>> sigma = np.array([[0.04, 0.0], [0.0, 0.01]])
    >>> rc_pct = risk_contribution(w, sigma, pct=True)
    >>> np.allclose(rc_pct, [0.8, 0.2])
    True

    """
    w = np.asarray(w, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square 2-D covariance matrix.")
    if sigma.shape[0] != w.shape[0]:
        raise ValueError(
            f"sigma shape {sigma.shape} does not match w shape {w.shape}."
        )
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma contains non-finite values (NaN or inf).")
    if not np.all(np.isfinite(w)):
        raise ValueError("w contains non-finite values (NaN or inf).")

    mr = marginal_risk(w, sigma)
    rc = w * mr

    if not pct:
        return rc

    # Normalize to sum to 1
    sigma_w = sigma @ w
    sigma_p = np.sqrt(w @ sigma_w)

    if sigma_p == 0.0:
        return np.zeros_like(w)

    return rc / sigma_p


def roll_risk_contribution(
    W: NDArray,
    X: NDArray,
    n: int = 252,
    cov: Callable[[NDArray], NDArray] | None = None,
    pct: bool = True,
) -> NDArray:
    r""" Causal rolling risk contributions over a time series.

    For each time step :math:`t \geq n`, estimate the covariance matrix
    from returns in the past :math:`n` periods and compute risk
    contributions for the weights at time :math:`t`. Earlier rows are
    filled with NaN (no covariance estimate yet).

    Parameters
    ----------
    W : array_like
        Weight time series, shape ``(T, N)``.
    X : array_like
        Returns panel, shape ``(T, N)``, rows in chronological order
        (oldest first).
    n : int, optional
        Window length for covariance estimation. Default 252.
    cov : callable, optional
        Covariance estimator callable that accepts a returns panel and
        returns a symmetric ``(N, N)`` matrix. If None (default), uses
        :func:`numpy.cov` with ``rowvar=False``.
    pct : bool, optional
        If True (default), return percentage contributions summing to 1.
        If False, return absolute contributions summing to portfolio
        volatility per period.

    Returns
    -------
    np.ndarray
        Risk contributions per period, shape ``(T, N)``. Rows ``t < n``
        are filled with NaN.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 3))
    >>> W = rng.uniform(0.2, 0.4, (50, 3))
    >>> W /= W.sum(axis=1, keepdims=True)
    >>> rc = roll_risk_contribution(W, X, n=20)
    >>> rc.shape
    (50, 3)
    >>> np.isnan(rc[:20]).all()
    True

    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if W.ndim != 2 or X.ndim != 2:
        raise ValueError("W and X must both be 2-D arrays.")
    if W.shape != X.shape:
        raise ValueError(
            f"W shape {W.shape} does not match X shape {X.shape}."
        )
    if n < 1 or n >= W.shape[0]:
        raise ValueError(
            f"Window length n={n} must be >= 1 and < T={W.shape[0]}."
        )
    if not np.all(np.isfinite(W)):
        raise ValueError("W contains non-finite values (NaN or inf).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or inf).")

    T, N = W.shape
    rc = np.full((T, N), np.nan, dtype=np.float64)

    for t in range(n, T):
        window = X[t - n : t]

        # Estimate covariance: use provided callable or numpy.cov
        if cov is not None:
            sigma_t = cov(window)
        else:
            sigma_t = np.atleast_2d(np.cov(window, rowvar=False))

        # Compute risk contribution for weights at time t
        rc[t] = risk_contribution(W[t], sigma_t, pct=pct)

    return rc
