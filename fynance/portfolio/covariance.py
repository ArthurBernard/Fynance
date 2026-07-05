#!/usr/bin/env python3
# coding: utf-8

r""" Conditioned covariance estimators for portfolio allocation.

Estimators of the asset return covariance matrix, from the raw sample
estimator to shrinkage, exponentially-weighted and denoising variants
that are better conditioned for use in Markowitz-style optimizers
(minimum variance, ERC, MDP, ...). Every estimator takes a ``(T, N)``
array of returns (rows = time, columns = assets) and returns a
symmetric ``(N, N)`` covariance matrix, so they are interchangeable as
the ``cov=`` callable consumed by the allocators.

Main entry points
-----------------
- :func:`sample_cov` — plain sample covariance (``numpy.cov`` wrapper).
- :func:`ledoit_wolf` — closed-form linear shrinkage (identity,
  constant-correlation or diagonal target).
- :func:`ewma_cov` — RiskMetrics-style exponentially weighted
  covariance.
- :func:`factor_cov` — low-rank + diagonal (statistical factor model)
  covariance.
- :func:`denoise_cov` — Marchenko-Pastur eigenvalue clipping on the
  correlation matrix.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ['sample_cov', 'ledoit_wolf', 'ewma_cov', 'factor_cov', 'denoise_cov']


# =========================================================================== #
#                                helpers                                      #
# =========================================================================== #


def _validate_returns(X: NDArray) -> NDArray:
    """ Cast to float64, reshape 1-D input to a single column, check finiteness.

    Parameters
    ----------
    X : array_like
        Returns panel, 1-D (single asset) or 2-D (``T, N``).

    Returns
    -------
    np.ndarray
        Validated ``(T, N)`` float64 array.

    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got ndim={X.ndim}.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or inf).")

    return X


def _symmetrize(M: NDArray) -> NDArray:
    """ Force exact symmetry of a square matrix: ``(M + M.T) / 2``. """
    M = np.asarray(M, dtype=np.float64)

    return (M + M.T) / 2.0


# =========================================================================== #
#                                sample covariance                            #
# =========================================================================== #


def sample_cov(X: NDArray, ddof: int = 1) -> NDArray:
    r""" Sample covariance matrix of a returns panel.

    Thin, validated wrapper around :func:`numpy.cov`: a 1-D input is
    treated as a single asset (returns ``[[var]]``); non-finite values
    raise.

    Parameters
    ----------
    X : array_like
        Returns panel, shape ``(T,)`` or ``(T, N)``.
    ddof : int, optional
        Delta degrees of freedom (``ddof=1`` is the unbiased sample
        covariance, ``ddof=0`` the population/maximum-likelihood one).
        Default 1.

    Returns
    -------
    np.ndarray
        Symmetric ``(N, N)`` covariance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    >>> sample_cov(X, ddof=1)
    array([[1.        , 1.        ],
           [1.        , 2.33333333]])

    """
    X = _validate_returns(X)
    cov = np.atleast_2d(np.cov(X, rowvar=False, ddof=ddof))

    return _symmetrize(cov)


# =========================================================================== #
#                                Ledoit-Wolf shrinkage                        #
# =========================================================================== #


def _lw_shrinkage(X: NDArray, target: str) -> tuple[float, NDArray, NDArray]:
    """ Ledoit-Wolf shrinkage intensity, sample covariance and target matrix.

    Shared computation for :func:`ledoit_wolf` and :func:`_lw_intensity`.
    Implements the closed-form intensity of Ledoit & Wolf (2004a/2004b):
    ``kappa_hat = (pi_hat - rho_hat) / gamma_hat``, ``delta = clip(kappa_hat
    / T, 0, 1)``, where ``pi_hat`` is the (target-independent) sum of
    asymptotic variances of the sample covariance entries, ``rho_hat`` the
    asymptotic covariance between the target and the sample covariance, and
    ``gamma_hat = ||F - S||_F^2`` the (squared) misspecification of the
    target.

    """
    if target not in ('identity', 'const_corr', 'diag'):
        raise ValueError(
            f"Unknown target {target!r}; expected 'identity', 'const_corr' "
            "or 'diag'."
        )

    X = _validate_returns(X)
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T

    if N == 1:
        return 0.0, S, S.copy()

    X2 = Xc ** 2
    row_sq = X2.sum(axis=1)
    pi_hat = float(np.mean(row_sq ** 2) - np.sum(S ** 2))
    pi_ii = np.mean(X2 ** 2, axis=0) - np.diag(S) ** 2

    if target == 'identity':
        mu = np.trace(S) / N
        F = mu * np.eye(N)
        rho_hat = float((np.mean(row_sq ** 2) - (mu * N) ** 2) / N)
    elif target == 'diag':
        F = np.diag(np.diag(S))
        rho_hat = float(np.sum(pi_ii))
    else:  # const_corr
        d = np.sqrt(np.diag(S))
        denom = np.outer(d, d)
        safe_denom = np.where(denom > 0, denom, 1.0)
        corr = np.where(denom > 0, S / safe_denom, 0.0)
        r_bar = (np.sum(corr) - N) / (N * (N - 1))

        F = r_bar * denom
        np.fill_diagonal(F, np.diag(S))

        Z = X2 - np.diag(S)[None, :]
        G = Z * Xc
        theta1 = G.T @ Xc / T  # theta1[i, j] = theta_hat_{ii,ij}
        safe_d = np.where(d > 0, d, 1.0)
        ratio = np.where(
            (d[:, None] > 0) & (d[None, :] > 0),
            d[None, :] / safe_d[:, None],
            0.0,
        )
        term = ratio * theta1 + ratio.T * theta1.T
        np.fill_diagonal(term, 0.0)
        rho_hat = float(np.sum(pi_ii) + 0.5 * r_bar * np.sum(term))

    gamma_hat = float(np.sum((F - S) ** 2))
    if gamma_hat <= 0.0:
        delta = 0.0
    else:
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa_hat / T, 0.0, 1.0))

    return delta, S, F


def _lw_intensity(X: NDArray, target: str = 'const_corr') -> float:
    """ Ledoit-Wolf shrinkage intensity (in ``[0, 1]``) for a returns panel.

    Exposed as a private helper so callers/tests can inspect the intensity
    without recomputing the shrunk matrix. See :func:`ledoit_wolf`.

    """
    delta, _, _ = _lw_shrinkage(X, target)

    return delta


def ledoit_wolf(X: NDArray, target: str = 'const_corr') -> NDArray:
    r""" Ledoit-Wolf linear shrinkage covariance estimator.

    Closed-form convex combination :math:`(1 - \delta) S + \delta F` of the
    sample covariance :math:`S` and a low-variance target :math:`F`, with
    the shrinkage intensity :math:`\delta \in [0, 1]` chosen to minimize
    the asymptotic expected Frobenius loss. Well conditioned even when
    :math:`N` is comparable to or larger than :math:`T` (where the plain
    sample covariance is singular or ill-conditioned).

    Parameters
    ----------
    X : array_like
        Returns panel, shape ``(T,)`` or ``(T, N)``.
    target : {'const_corr', 'identity', 'diag'}, optional
        Shrinkage target:

        - ``'const_corr'`` — constant-correlation matrix built from the
          mean off-diagonal correlation (Ledoit & Wolf, 2004b).
        - ``'identity'`` — mean-variance scaled identity (Ledoit & Wolf,
          2004a).
        - ``'diag'`` — ``diag(S)``, i.e. shrink off-diagonal entries only.

        Default ``'const_corr'``.

    Returns
    -------
    np.ndarray
        Symmetric ``(N, N)`` shrunk covariance matrix.

    References
    ----------
    .. [1] O. Ledoit, M. Wolf, "A well-conditioned estimator for
       large-dimensional covariance matrices", Journal of Multivariate
       Analysis, 88(2), 2004, 365-411.
    .. [2] O. Ledoit, M. Wolf, "Honey, I shrunk the sample covariance
       matrix", The Journal of Portfolio Management, 30(4), 2004, 110-119.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 4))
    >>> S = ledoit_wolf(X)
    >>> S.shape
    (4, 4)
    >>> bool(np.allclose(S, S.T))
    True

    """
    delta, S, F = _lw_shrinkage(X, target)
    cov = delta * F + (1.0 - delta) * S

    return _symmetrize(cov)


# =========================================================================== #
#                       exponentially weighted covariance                     #
# =========================================================================== #


@njit(cache=True)
def _ewma_cov_kernel(Xc: NDArray, lam: float) -> NDArray:
    """ Accumulate lambda-weighted outer products of demeaned rows.

    ``Xc`` is already demeaned; weights ``lam ** (T - 1 - t)`` are
    normalized to sum to one inside the loop, no ``(T, N, N)`` temp.

    """
    T, N = Xc.shape
    weights = np.empty(T, dtype=np.float64)
    wsum = 0.0
    for t in range(T):
        wt = lam ** (T - 1 - t)
        weights[t] = wt
        wsum += wt

    cov = np.zeros((N, N), dtype=np.float64)
    for t in range(T):
        wn = weights[t] / wsum
        for i in range(N):
            wxi = wn * Xc[t, i]
            for j in range(i, N):
                cov[i, j] += wxi * Xc[t, j]

    for i in range(N):
        for j in range(i + 1, N):
            cov[j, i] = cov[i, j]

    return cov


def ewma_cov(X: NDArray, halflife: float = 63.0) -> NDArray:
    r""" RiskMetrics-style exponentially weighted covariance matrix.

    Weights recent observations more heavily: :math:`\lambda = 0.5^{1 /
    halflife}`, weight of observation :math:`t` (out of :math:`T`, most
    recent last) proportional to :math:`\lambda^{T - 1 - t}`, normalized to
    sum to one. Data is demeaned by the plain (equally-weighted) column
    mean before accumulating weighted outer products.

    Parameters
    ----------
    X : array_like
        Returns panel, shape ``(T,)`` or ``(T, N)``, rows in chronological
        order (oldest first).
    halflife : float, optional
        Number of steps after which a past observation's weight is halved.
        Default 63.0 (~ one quarter of trading days).

    Returns
    -------
    np.ndarray
        Symmetric ``(N, N)`` covariance matrix.

    References
    ----------
    .. [1] J.P. Morgan/Reuters, "RiskMetrics -- Technical Document",
       4th edition, 1996.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 3))
    >>> S = ewma_cov(X, halflife=20.0)
    >>> S.shape
    (3, 3)
    >>> bool(np.allclose(S, S.T))
    True

    """
    X = _validate_returns(X)
    if halflife <= 0:
        raise ValueError("halflife must be strictly positive.")

    Xc = X - X.mean(axis=0)
    lam = 0.5 ** (1.0 / halflife)
    cov = _ewma_cov_kernel(Xc, lam)

    return _symmetrize(cov)


# =========================================================================== #
#                                factor covariance                            #
# =========================================================================== #


def factor_cov(X: NDArray, n_factors: int = 3) -> NDArray:
    r""" Statistical factor-model covariance (low-rank + diagonal).

    Eigendecomposes the (population) sample covariance and keeps the
    ``k = min(n_factors, N)`` largest eigenpairs as a common (systematic)
    component :math:`B B^\top` with loadings :math:`B = V_k
    \text{diag}(\sqrt{l_k})`; the remainder of each asset's variance is
    kept as an idiosyncratic diagonal term. The result is positive
    semi-definite by construction and matches the sample covariance's
    diagonal (total variance) exactly.

    Parameters
    ----------
    X : array_like
        Returns panel, shape ``(T,)`` or ``(T, N)``.
    n_factors : int, optional
        Number of factors to keep, clipped to ``N``. Default 3.

    Returns
    -------
    np.ndarray
        Symmetric ``(N, N)`` covariance matrix, ``B @ B.T + diag(idio)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 5))
    >>> S = factor_cov(X, n_factors=2)
    >>> S.shape
    (5, 5)
    >>> bool(np.all(np.linalg.eigvalsh(S) >= -1e-10))
    True

    """
    if n_factors < 1:
        raise ValueError("n_factors must be a positive integer.")

    X = _validate_returns(X)
    T, N = X.shape
    S = sample_cov(X, ddof=0)
    k = min(n_factors, N)

    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1][:k]
    lk = np.clip(eigvals[order], 0.0, None)
    Vk = eigvecs[:, order]
    B = Vk * np.sqrt(lk)[None, :]

    common = B @ B.T
    eps = 1e-12 * np.trace(S) / N
    idio = np.clip(np.diag(S) - np.diag(common), eps, None)
    cov = common + np.diag(idio)

    return _symmetrize(cov)


# =========================================================================== #
#                          Marchenko-Pastur denoising                         #
# =========================================================================== #


def denoise_cov(sigma: NDArray, n_obs: int, method: str = 'clip') -> NDArray:
    r""" Marchenko-Pastur eigenvalue clipping on the correlation matrix.

    Converts ``sigma`` to a correlation matrix, replaces the "noise"
    eigenvalues (those at or below the Marchenko-Pastur upper edge
    :math:`\lambda_+ = (1 + \sqrt{q})^2`, :math:`q = N / n\_obs`) by their
    mean (trace-preserving), reconstructs, forces a unit diagonal and
    rescales back by the original volatilities.

    Parameters
    ----------
    sigma : array_like
        Symmetric ``(N, N)`` covariance matrix to denoise.
    n_obs : int
        Number of observations the covariance was estimated on.
    method : {'clip'}, optional
        Denoising method. Only ``'clip'`` (constant-residual-eigenvalue) is
        implemented; the keyword is validated so the signature is stable
        for future methods. Default ``'clip'``.

    Returns
    -------
    np.ndarray
        Symmetric ``(N, N)`` denoised covariance matrix, same diagonal and
        trace as ``sigma``.

    References
    ----------
    .. [1] V.A. Marchenko, L.A. Pastur, "Distribution of eigenvalues for
       some sets of random matrices", Mat. Sb., 72(114), 1967, 507-536.
    .. [2] M. Lopez de Prado, "Machine Learning for Asset Managers",
       Cambridge University Press, 2020 (constant residual eigenvalue
       method).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((60, 10))
    >>> S = sample_cov(X, ddof=0)
    >>> D = denoise_cov(S, n_obs=60)
    >>> D.shape
    (10, 10)
    >>> bool(np.allclose(np.diag(D), np.diag(S)))
    True

    """
    if method != 'clip':
        raise ValueError(f"Unknown method {method!r}; only 'clip' is supported.")
    if n_obs <= 0:
        raise ValueError("n_obs must be strictly positive.")

    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square 2-D covariance matrix.")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma contains non-finite values (NaN or inf).")

    sigma = _symmetrize(sigma)
    N = sigma.shape[0]
    vols = np.sqrt(np.diag(sigma))
    denom = np.outer(vols, vols)
    safe_denom = np.where(denom > 0, denom, 1.0)
    corr = np.where(denom > 0, sigma / safe_denom, 0.0)
    np.fill_diagonal(corr, 1.0)

    q = N / float(n_obs)
    lam_plus = (1.0 + np.sqrt(q)) ** 2

    eigvals, eigvecs = np.linalg.eigh(corr)
    mask = eigvals <= lam_plus
    if np.any(mask):
        eigvals = np.where(mask, np.mean(eigvals[mask]), eigvals)

    denoised_corr = (eigvecs * eigvals) @ eigvecs.T
    np.fill_diagonal(denoised_corr, 1.0)
    cov = denoised_corr * denom

    return _symmetrize(cov)
