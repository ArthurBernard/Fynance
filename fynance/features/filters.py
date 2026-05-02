#!/usr/bin/env python3
# coding: utf-8

""" Filter functions for time-series financial data. """

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

__all__ = ['kalman_filter', 'rts_smoother', 'kalman_loglikelihood', 'fit_kalman']


def kalman_filter(
    y: NDArray,
    G: NDArray,
    F: NDArray,
    W: NDArray,
    V: NDArray,
    m0: NDArray | None = None,
    C0: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """ Linear Gaussian Kalman filter.

    State-space model::

        x_t = G @ x_{t-1} + w_t,   w_t ~ N(0, W)
        y_t = F @ x_t    + v_t,   v_t ~ N(0, V)

    Alternates between a *predict* step (prior ``a``, ``R``) and an *update*
    step (posterior ``m``, ``C``).

    Parameters
    ----------
    y : np.ndarray of shape (T, n)
        Observed data.
    G : np.ndarray of shape (n, n)
        State transition matrix.
    F : np.ndarray of shape (n, n)
        Observation matrix.
    W : np.ndarray of shape (n, n)
        Process noise covariance.
    V : np.ndarray of shape (n, n)
        Observation noise covariance.
    m0 : np.ndarray of shape (n,), optional
        Initial state mean (default: zeros).
    C0 : np.ndarray of shape (n, n), optional
        Initial state covariance (default: identity).

    Returns
    -------
    m : np.ndarray of shape (T, n)
        Filtered state means (posterior).
    C : np.ndarray of shape (T, n, n)
        Filtered state covariances (posterior).
    a : np.ndarray of shape (T, n)
        Prior state means (predict step).
    R : np.ndarray of shape (T, n, n)
        Prior state covariances (predict step).
    e : np.ndarray of shape (T, n)
        Innovations ``y_t - F @ a_t``.
    S : np.ndarray of shape (T, n, n)
        Innovation covariances ``F @ R_t @ F.T + V``.

    References
    ----------
    R. E. Kalman, "A New Approach to Linear Filtering and Prediction
    Problems", Journal of Basic Engineering, 1960.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal((5, 1))
    >>> G = F = W = V = np.eye(1)
    >>> m, C, a, R, e, S = kalman_filter(y, G, F, W, V)
    >>> m.shape
    (5, 1)
    >>> C.shape
    (5, 1, 1)
    >>> e.shape
    (5, 1)

    """
    y = np.asarray(y, dtype=float)
    G = np.asarray(G, dtype=float)
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)
    V = np.asarray(V, dtype=float)

    T, n = y.shape

    m = np.zeros((T, n))
    C = np.zeros((T, n, n))
    a = np.zeros((T, n))
    R = np.zeros((T, n, n))
    e = np.zeros((T, n))
    S = np.zeros((T, n, n))

    m_prev = np.zeros(n) if m0 is None else np.asarray(m0, dtype=float)
    C_prev = np.eye(n) if C0 is None else np.asarray(C0, dtype=float)

    for t in range(T):
        # Predict
        a[t] = G @ m_prev
        R[t] = G @ C_prev @ G.T + W

        # Innovation
        e[t] = y[t] - F @ a[t]
        S[t] = F @ R[t] @ F.T + V

        # Update
        K = R[t] @ F.T @ np.linalg.inv(S[t])
        m[t] = a[t] + K @ e[t]
        C[t] = (np.eye(n) - K @ F) @ R[t]

        m_prev = m[t]
        C_prev = C[t]

    return m, C, a, R, e, S


def rts_smoother(
    m: NDArray,
    C: NDArray,
    a: NDArray,
    R: NDArray,
    G: NDArray,
) -> tuple[NDArray, NDArray]:
    """ Rauch-Tung-Striebel (RTS) smoother.

    Backward pass over the output of :func:`kalman_filter`.  Produces
    smoothed estimates that use all observations, not just past ones.

    Parameters
    ----------
    m : np.ndarray of shape (T, n)
        Filtered means from :func:`kalman_filter`.
    C : np.ndarray of shape (T, n, n)
        Filtered covariances from :func:`kalman_filter`.
    a : np.ndarray of shape (T, n)
        Prior means from :func:`kalman_filter`.
    R : np.ndarray of shape (T, n, n)
        Prior covariances from :func:`kalman_filter`.
    G : np.ndarray of shape (n, n)
        State transition matrix (same as passed to :func:`kalman_filter`).

    Returns
    -------
    ms : np.ndarray of shape (T, n)
        Smoothed state means.
    Cs : np.ndarray of shape (T, n, n)
        Smoothed state covariances.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal((5, 1))
    >>> G = F = W = V = np.eye(1)
    >>> m, C, a, R, e, S = kalman_filter(y, G, F, W, V)
    >>> ms, Cs = rts_smoother(m, C, a, R, G)
    >>> ms.shape
    (5, 1)
    >>> bool((np.abs(ms) <= np.abs(y) * 2).all())
    True

    """
    T, n = m.shape
    ms = m.copy()
    Cs = C.copy()

    for t in range(T - 2, -1, -1):
        L = C[t] @ G.T @ np.linalg.pinv(R[t + 1])
        ms[t] = m[t] + L @ (ms[t + 1] - a[t + 1])
        Cs[t] = C[t] + L @ (Cs[t + 1] - R[t + 1]) @ L.T

    return ms, Cs


def kalman_loglikelihood(e: NDArray, S: NDArray) -> float:
    """ Prediction-error decomposition log-likelihood.

    Computes::

        L = -0.5 * sum_t [ log|S_t| + e_t^T S_t^{-1} e_t + n log(2π) ]

    Parameters
    ----------
    e : np.ndarray of shape (T, n)
        Innovations from :func:`kalman_filter`.
    S : np.ndarray of shape (T, n, n)
        Innovation covariances from :func:`kalman_filter`.

    Returns
    -------
    float
        Log-likelihood value (higher is better).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal((20, 1))
    >>> G = F = W = V = np.eye(1)
    >>> m, C, a, R, e, S = kalman_filter(y, G, F, W, V)
    >>> ll = kalman_loglikelihood(e, S)
    >>> isinstance(ll, float)
    True
    >>> ll < 0
    True

    """
    T, n = e.shape
    ll = 0.0
    log2pi = math.log(2 * math.pi)
    for t in range(T):
        sign, logdet = np.linalg.slogdet(S[t])
        if sign <= 0:
            return -np.inf
        quad = float(e[t] @ np.linalg.inv(S[t]) @ e[t])
        ll -= 0.5 * (logdet + quad + n * log2pi)
    return ll


def fit_kalman(
    y: NDArray,
    G: NDArray,
    F: NDArray,
    *,
    fit_W: bool = True,
    fit_V: bool = True,
    W0: NDArray | None = None,
    V0: NDArray | None = None,
    m0: NDArray | None = None,
    C0: NDArray | None = None,
) -> dict:
    """ Estimate Kalman filter noise covariances by maximum likelihood.

    Optimises W and/or V via the prediction-error log-likelihood computed by
    :func:`kalman_loglikelihood`.  Parameters are searched in log-space to
    guarantee positive-definiteness for diagonal matrices.

    Only diagonal W and V are supported for estimation.  Pass fixed
    non-diagonal matrices through ``W0`` / ``V0`` with ``fit_W=False`` /
    ``fit_V=False``.

    Parameters
    ----------
    y : np.ndarray of shape (T, n)
        Observed data.
    G : np.ndarray of shape (n, n)
        State transition matrix (fixed).
    F : np.ndarray of shape (n, n)
        Observation matrix (fixed).
    fit_W : bool, optional
        Whether to estimate W (default True).
    fit_V : bool, optional
        Whether to estimate V (default True).
    W0 : np.ndarray of shape (n, n), optional
        Initial / fixed W.  If ``fit_W=False`` this value is used as-is.
        Defaults to identity.
    V0 : np.ndarray of shape (n, n), optional
        Initial / fixed V.  If ``fit_V=False`` this value is used as-is.
        Defaults to identity.
    m0 : np.ndarray of shape (n,), optional
        Initial state mean (default: zeros).
    C0 : np.ndarray of shape (n, n), optional
        Initial state covariance (default: identity).

    Returns
    -------
    dict with keys:
        - ``'W'`` : np.ndarray — estimated (or fixed) process noise covariance.
        - ``'V'`` : np.ndarray — estimated (or fixed) observation noise covariance.
        - ``'loglik'`` : float — log-likelihood at the optimum.
        - ``'success'`` : bool — whether the optimizer converged.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n, T = 1, 200
    >>> W_true = np.eye(n) * 0.5
    >>> V_true = np.eye(n) * 1.0
    >>> x = np.cumsum(rng.multivariate_normal(np.zeros(n), W_true, T), axis=0)
    >>> y = x + rng.multivariate_normal(np.zeros(n), V_true, T)
    >>> G = F = np.eye(n)
    >>> result = fit_kalman(y, G, F)
    >>> result['success']
    True
    >>> result['W'].shape
    (1, 1)

    """
    y = np.asarray(y, dtype=float)
    G = np.asarray(G, dtype=float)
    F = np.asarray(F, dtype=float)
    n = y.shape[1]

    W_fixed = (np.eye(n) if W0 is None else np.asarray(W0, dtype=float))
    V_fixed = (np.eye(n) if V0 is None else np.asarray(V0, dtype=float))

    # Build initial log-space parameter vector
    x0_parts = []
    if fit_W:
        x0_parts.append(np.log(np.diag(W_fixed)))
    if fit_V:
        x0_parts.append(np.log(np.diag(V_fixed)))

    if not x0_parts:
        _, _, _, _, e, S = kalman_filter(y, G, F, W_fixed, V_fixed, m0, C0)
        return {
            'W': W_fixed,
            'V': V_fixed,
            'loglik': kalman_loglikelihood(e, S),
            'success': True,
        }

    x0 = np.concatenate(x0_parts)

    def _neg_loglik(params):
        idx = 0
        if fit_W:
            W = np.diag(np.exp(params[idx: idx + n]))
            idx += n
        else:
            W = W_fixed
        V = np.diag(np.exp(params[idx: idx + n])) if fit_V else V_fixed
        _, _, _, _, e, S = kalman_filter(y, G, F, W, V, m0, C0)
        return -kalman_loglikelihood(e, S)

    res = minimize(_neg_loglik, x0, method='L-BFGS-B')

    idx = 0
    if fit_W:
        W_hat = np.diag(np.exp(res.x[idx: idx + n]))
        idx += n
    else:
        W_hat = W_fixed
    V_hat = np.diag(np.exp(res.x[idx: idx + n])) if fit_V else V_fixed

    _, _, _, _, e, S = kalman_filter(y, G, F, W_hat, V_hat, m0, C0)

    return {
        'W': W_hat,
        'V': V_hat,
        'loglik': kalman_loglikelihood(e, S),
        'success': bool(res.success),
    }
