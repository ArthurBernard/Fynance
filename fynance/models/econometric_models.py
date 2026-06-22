#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-20 10:39:56
# @Last modified by: ArthurBernard
# @Last modified time: 2019-05-23 17:31:40

""" Classical econometric time-series models.

Pure-NumPy implementations of moving-average, autoregressive and
GARCH-family processes used to simulate or fit return series. Parameter
estimation is delegated to the Numba routines in
:mod:`fynance.estimator` via :func:`get_parameters`, a single
maximum-likelihood entry point for all four model types.

Main entry points
-----------------
- :func:`MA` — Moving Average process.
- :func:`ARMA` — AutoRegressive Moving Average.
- :func:`ARMA_GARCH` — ARMA with GARCH(p, q) conditional variance.
- :func:`ARMAX_GARCH` — ARMA-GARCH with exogenous regressors.
- :func:`get_parameters` — fit the parameters of any of the above.

"""

# Built-in packages

# External packages
import numpy as np
import polars as pl

# Local packages
from numba import njit


@njit(cache=True)
def _ma(y, theta, c, q):
    """ MA(q) residual recursion (numba kernel). """
    T = y.size
    u = np.zeros(T)
    for t in range(T):
        s = 0.0
        for i in range(min(t, q)):
            s += u[t - i - 1] * theta[i]
        if s > 1e12 or s < -1e12:
            return 1e6 * np.ones(T)
        u[t] = y[t] - c - s
    return u


@njit(cache=True)
def _arma(y, phi, theta, c, p, q):
    """ ARMA(p, q) residual recursion (numba kernel). """
    T = y.size
    u = np.zeros(T)
    for t in range(T):
        s = 0.0
        for i in range(min(t, max(q, p))):
            if i < q:
                s += u[t - i - 1] * theta[i]
            if i < p:
                s += y[t - i - 1] * phi[i]
            if s > 1e12 or s < -1e12:
                return 1e6 * np.ones(T)
        u[t] = y[t] - c - s
    return u


@njit(cache=True)
def _arma_garch(y, phi, theta, alpha, beta, c, omega, p, q, Q, P):
    """ ARMA(p, q)-GARCH(Q, P) residual/volatility recursion (numba kernel). """
    T = y.size
    u = np.zeros(T)
    h = np.zeros(T)
    for t in range(T):
        arma = 0.0
        arch = 0.0
        for i in range(min(t, max(max(q, p), max(Q, P)))):
            if i < p:
                arma += y[t - i - 1] * phi[i]
            if i < q:
                arma += u[t - i - 1] * theta[i]
            if i < Q:
                arch += u[t - i - 1] ** 2 * alpha[i]
            if i < P:
                arch += h[t - i - 1] ** 2 * beta[i]
            if arch < 0.0:
                return 1e8 * np.ones(T), np.ones(T)
            if arch > 1e12 or arma > 1e12 or arma < -1e12:
                return 1e6 * np.ones(T), np.ones(T)
        u[t] = y[t] - c - arma
        h[t] = np.sqrt(omega + arch)
    return u, h


@njit(cache=True)
def _armax_garch(y, x, phi, psi, theta, alpha, beta, c, omega, p, q, Q, P):
    """ ARMAX(p, q)-GARCH(Q, P) residual/volatility recursion (numba kernel). """
    T = y.size
    u = np.zeros(T)
    h = np.zeros(T)
    for t in range(T):
        armax = 0.0
        for k in range(x.shape[1]):
            armax += x[t, k] * psi[k]
        arch = 0.0
        for i in range(min(t, max(max(q, p), max(Q, P)))):
            if i < p:
                armax += y[t - i - 1] * phi[i]
            if i < q:
                armax += u[t - i - 1] * theta[i]
            if i < Q:
                arch += u[t - i - 1] ** 2 * alpha[i]
            if i < P:
                arch += h[t - i - 1] ** 2 * beta[i]
            if arch < 0.0:
                return 1e8 * np.ones(T), np.ones(T)
            if arch > 1e12 or armax > 1e12 or armax < -1e12:
                return 1e6 * np.ones(T), np.ones(T)
        u[t] = y[t] - c - armax
        h[t] = np.sqrt(omega + arch)
    return u, h

__all__ = [
    'get_parameters', 'MA', 'ARMA', 'ARMA_GARCH', 'ARMAX_GARCH'
]

# =========================================================================== #
#                             PARAMETERS FUNCTION                             #
# =========================================================================== #


def get_parameters(params, p=0, q=0, Q=0, P=0, cons=True):
    """ Get parameters for ARMA-GARCH models.

    Helper that splits the flat ``params`` vector returned by
    maximum-likelihood estimation into the structured groups expected
    by the model evaluation routines: AR coefficients ``phi``, MA
    coefficients ``theta``, GARCH ARCH/GARCH coefficients
    ``alpha``/``beta``, and constants ``c`` and ``omega``. Pass the
    same ``p, q, Q, P, cons`` configuration that was used at the
    estimation step (see :mod:`fynance.estimator`).

    Parameters
    ----------
    params : np.ndarray[np.float64, ndim=1]
        Array of model parameters.
    p, q, Q, P : int, optional
        Order of model, default is 0.
    cons : bool, optional
        True if model contains constant, default is True.

    Returns
    -------
    phi : np.ndarray[np.float64, ndim=1]
        AR parameters.
    theta : np.ndarray[np.float64, ndim=1]
        MA parameters.
    alpha : np.ndarray[np.float64, ndim=1]
        First part GARCH parameters.
    beta : np.ndarray[np.float64, ndim=1]
        Last part GARCH parameters.
    c : float
        Constant of ARMA part.
    omega : float
        Constants of GARCH part.

    See Also
    --------
    ARMAX_GARCH, ARMA_GARCH, ARMA, MA.

    """
    i = 0
    if cons:
        c = params[i]
        i += 1
    else:
        c = 0.
    if p > 0:
        phi = params[i: p + i]
        i += p
    else:
        phi = np.array([0.], dtype=np.float64)
    if q > 0:
        theta = params[i: q + i]
        i += q
    else:
        theta = np.array([0.], dtype=np.float64)
    if Q > 0 or P > 0:
        omega = params[i]
        i += 1
        if Q > 0:
            alpha = params[i: Q + i]
            i += Q
        else:
            alpha = np.array([0.], dtype=np.float64)
        if P > 0:
            beta = params[i: P + i]
            i += P
        else:
            beta = np.array([0.], dtype=np.float64)
    else:
        omega = 0.
        alpha = np.array([0.], dtype=np.float64)
        beta = np.array([0.], dtype=np.float64)
    return phi, theta, alpha, beta, c, omega


# =========================================================================== #
#                                   MODELs                                    #
# =========================================================================== #


def MA(y, theta, c, q):
    r""" Moving Average model of order `q` s.t:

    .. math:: y_t = c + \theta_1 * u_{t-1} + ... + \theta_q * u_{t-q} + u_t

    Parameters
    ----------
    y : np.ndarray[np.float64, ndim=1]
        Time series.
    theta : np.ndarray[np.float64, ndim=1]
        Coefficients of model.
    c : np.float64
        Constant of the model.
    q : int
        Order of MA(q) model.

    Returns
    -------
    u : np.ndarray[ndim=1, dtype=np.float64]
        Residual of the model.

    Examples
    --------
    >>> y = np.array([3, 4, 6, 8, 5, 3])
    >>> MA(y=y, theta=np.array([0.8]), c=3, q=1)
    array([ 0.    ,  1.    ,  2.2   ,  3.24  , -0.592 ,  0.4736])

    See Also
    --------
    ARMA_GARCH, ARMA, ARMAX_GARCH

    """
    # Set type of variables
    if isinstance(y, (pl.DataFrame, pl.Series)):
        y = y.to_numpy()

    elif isinstance(y, list):
        y = np.asarray(y)

    y = y.astype(np.float64).reshape([y.size])

    if isinstance(theta, list):
        theta = np.asarray(theta)

    theta = theta.astype(np.float64).reshape([theta.size])
    # Compute residuals
    u = _ma(y, theta, float(c), int(q))

    return u


def ARMA(y, phi, theta, c, p, q):
    r""" AutoRegressive Moving Average model of order `q` and `p` s.t:

    .. math::

        y_t = c + \phi_1 * y_{t-1} + ... + \phi_p * y_{t-p}
        + \theta_1 * u_{t-1} + ... + \theta_q * u_{t-q} + u_t

    Parameters
    ----------
    y : np.ndarray[np.float64, ndim=1]
        Time series.
    phi : np.ndarray[np.float64, ndim=1]
        Coefficients of AR model.
    theta : np.ndarray[np.float64, ndim=1]
        Coefficients of MA model.
    c : np.float64
        Constant of the model.
    p : int
        Order of AR(p) model.
    q : int
        Order of MA(q) model.

    Returns
    -------
    u : np.ndarray[np.float64, ndim=1]
        Residual of the model.

    See Also
    --------
    ARMA_GARCH, ARMAX_GARCH, MA.

    """
    # Set type variables and parameters
    y = np.asarray(y, dtype=np.float64)
    y = y.reshape([y.size])
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    # Compute residuals
    u = _arma(y, phi, theta, float(c), int(p), int(q))

    return u


def ARMA_GARCH(y, phi, theta, alpha, beta, c, omega, p, q, Q, P):
    r""" AutoRegressive Moving Average model of order q and p, such that:

    .. math::

        y_t = c + \phi_1 * y_{t-1} + ... + \phi_p * y_{t-p}
        + \theta_1 * u_{t-1} + ... + \theta_q * u_{t-q} + u_t

    With Generalized AutoRegressive Conditional Heteroskedasticity volatility
    model of order `Q` and `P`, such that:

    .. math::
        u_t = z_t * h_t

        h_t^2 = \omega + \alpha_1 * u^2_{t-1} + ... + \alpha_Q * u^2_{t-Q}
        + \beta_1 * h^2_{t-1} + ... + \beta_P * h^2_{t-P}

    Parameters
    ----------
    y : np.ndarray[np.float64, ndim=1]
        Time series.
    phi : np.ndarray[np.float64, ndim=1]
        Coefficients of AR model.
    theta : np.ndarray[np.float64, ndim=1]
        Coefficients of MA model.
    alpha : np.ndarray[np.float64, ndim=1]
        Coefficients of MA part of GARCH.
    beta : np.ndarray[np.float64, ndim=1]
        Coefficients of AR part of GARCH.
    c : np.float64
        Constant of ARMA model.
    omega : np.float64
        Constant of GARCH model.
    p : int
        Order of AR(p) model.
    q : int
        Order of MA(q) model.
    Q : int
        Order of MA part of GARCH.
    P : int
        Order of AR part of GARCH.

    Returns
    -------
    u : np.ndarray[np.float64, ndim=1]
        Residual of the model.
    h : np.ndarray[np.float64, ndim=1]
        Conditional volatility of the model.

    See Also
    --------
    ARMAX_GARCH, ARMA, MA.

    """
    y = np.asarray(y, dtype=np.float64)
    y = y.reshape([y.size])
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    u, h = _arma_garch(
        y, phi, theta, alpha, beta, float(c), float(omega), int(p), int(q),
        int(Q), int(P)
    )

    return u, h


def ARMAX_GARCH(y, x, phi, psi, theta, alpha, beta, c, omega, p, q, Q, P):
    r""" AutoRegressive Moving Average model of order q and p, such that:

    .. math::

        y_t = c + \phi_1 * y_{t-1} + ... + \phi_p * y_{t-p}
        + \sum_k \psi_k * x_{t,k}
        + \theta_1 * u_{t-1} + ... + \theta_q * u_{t-q} + u_t

    With Generalized AutoRegressive Conditional Heteroskedasticity volatility
    model of order `Q` and `P`, such that:

    .. math::
        u_t = z_t * h_t

        h_t^2 = \omega + \alpha_1 * u^2_{t-1} + ... + \alpha_Q * u^2_{t-Q}
        + \beta_1 * h^2_{t-1} + ... + \beta_P * h^2_{t-P}

    Parameters
    ----------
    y : np.ndarray[np.float64, ndim=1]
        Time series.
    x : np.ndarray[np.float64, ndim=2]
        Time series of external features.
    phi : np.ndarray[np.float64, ndim=1]
        Coefficients of AR model.
    psi : np.ndarray[np.float64, ndim=1]
        Coefficients of external features.
    theta : np.ndarray[np.float64, ndim=1]
        Coefficients of MA model.
    alpha : np.ndarray[np.float64, ndim=1]
        Coefficients of MA part of GARCH.
    beta : np.ndarray[np.float64, ndim=1]
        Coefficients of AR part of GARCH.
    c : np.float64
        Constant of ARMA model.
    omega : np.float64
        Constant of GARCH model.
    p : int
        Order of AR(p) model.
    q : int
       Order of MA(q) model.
    Q : int
        Order of MA part of GARCH.
    P : int
        Order of AR part of GARCH.

    Returns
    -------
    u : np.ndarray[np.float64, ndim=1]
        Residual of the model.
    h : np.ndarray[np.float64, ndim=1]
        Conditional volatility of the model.

    See Also
    --------
    ARMA_GARCH, ARMA, MA.

    """
    # Set array variables
    y = np.asarray(y, dtype=np.float64)
    y = y.reshape([y.size])
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    # Compute residuals and volatility
    u, h = _armax_garch(
        y, x, phi, psi, theta, alpha, beta, float(c), float(omega), int(p),
        int(q), int(Q), int(P)
    )

    return u, h
