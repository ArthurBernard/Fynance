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
- :func:`loglik_garch` — GARCH-family (GARCH / GJR / EGARCH) log-likelihood
  under normal or standardized Student-t innovations, the objective for a
  maximum-likelihood fit driver.

"""

# Built-in packages
import math

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


@njit(cache=True)
def _gjr_garch(y, omega, alpha, gamma, beta):
    """ GJR-GARCH(1, 1) conditional std recursion (numba kernel).

    Variance recursion
    ``sigma_t^2 = omega + (alpha + gamma * 1[y_{t-1} < 0]) * y_{t-1}^2
    + beta * sigma_{t-1}^2`` with ``sigma_0^2 = omega`` (only the constant
    survives at t=0, matching the `_arma_garch` convention). ``y`` is used
    as the mean-zero innovation series. Returns the conditional standard
    deviation ``h_t = sigma_t``. With ``gamma == 0`` the ``h`` path is
    identical to the vanilla `_arma_garch` filter.
    """
    T = y.size
    h = np.zeros(T)
    h[0] = np.sqrt(omega)
    for t in range(1, T):
        ind = 1.0 if y[t - 1] < 0.0 else 0.0
        arch = (alpha + gamma * ind) * y[t - 1] ** 2 + beta * h[t - 1] ** 2
        h[t] = np.sqrt(omega + arch)
    return h


@njit(cache=True)
def _egarch(y, omega, alpha, gamma, beta, mean_abs_z):
    """ EGARCH(1, 1) conditional std recursion (numba kernel).

    Log-variance recursion
    ``ln sigma_t^2 = omega + beta * ln sigma_{t-1}^2
    + alpha * (|z_{t-1}| - mean_abs_z) + gamma * z_{t-1}`` with
    ``z_t = y_t / sigma_t`` and ``ln sigma_0^2 = omega`` (only the constant
    survives at t=0, matching the `_arma_garch` convention). ``mean_abs_z``
    is E|z| for the innovation distribution, supplied by the caller
    (``sqrt(2 / pi)`` for the normal, :func:`_mean_abs_standardized_t` for a
    standardized Student-t). Returns the conditional standard deviation
    ``h_t = sigma_t``.
    """
    T = y.size
    h = np.zeros(T)
    log_var = omega
    h[0] = np.exp(0.5 * log_var)
    for t in range(1, T):
        z = y[t - 1] / h[t - 1]
        log_var = (omega + beta * log_var
                   + alpha * (abs(z) - mean_abs_z) + gamma * z)
        h[t] = np.exp(0.5 * log_var)
    return h


@njit(cache=True)
def _mean_abs_standardized_t(nu):
    """ E|z| for a unit-variance Student-t (nu > 2) innovation (numba kernel).

    ``E|z| = 2 * sqrt(nu - 2) * Gamma((nu + 1) / 2)
    / ((nu - 1) * sqrt(pi) * Gamma(nu / 2))``.
    """
    ratio = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0))
    return 2.0 * math.sqrt(nu - 2.0) * ratio / ((nu - 1.0) * math.sqrt(math.pi))


@njit(cache=True)
def _loglik_normal(y, h):
    """ Gaussian log-likelihood given conditional std ``h`` (numba kernel). """
    T = y.size
    const = -0.5 * math.log(2.0 * math.pi)
    ll = 0.0
    for t in range(T):
        ll += const - math.log(h[t]) - 0.5 * (y[t] / h[t]) ** 2
    return ll


@njit(cache=True)
def _loglik_t(y, h, nu):
    """ Standardized Student-t (nu > 2) log-likelihood given ``h`` (numba). """
    T = y.size
    const = (math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0)
             - 0.5 * math.log(math.pi * (nu - 2.0)))
    ll = 0.0
    for t in range(T):
        z2 = (y[t] / h[t]) ** 2
        ll += (const - math.log(h[t])
               - 0.5 * (nu + 1.0) * math.log(1.0 + z2 / (nu - 2.0)))
    return ll


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


# =========================================================================== #
#                        GARCH-FAMILY LOG-LIKELIHOODS                         #
# =========================================================================== #


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def loglik_garch(
    params: np.ndarray,
    y: np.ndarray,
    model: str = 'garch',
    dist: str = 'normal',
) -> float:
    r""" Log-likelihood of a GARCH-family volatility model.

    Runs the conditional-variance recursion of the selected ``model`` over the
    mean-zero series ``y`` and returns the total log-likelihood of the
    innovations under the chosen ``dist``. The value is a *log-likelihood*
    (higher is better, to be **maximised**); invalid parameter regions map to
    ``-np.inf`` so a maximiser (or a minimiser of its negative) is repelled
    from them without exceptions being raised. Intended as the objective for a
    scipy maximum-likelihood driver (built in a later step).

    Parameters
    ----------
    params : np.ndarray[np.float64, ndim=1]
        Flat parameter vector, laid out per ``model`` (all variance
        parameters first, ``nu`` appended last when ``dist='t'``):

        - ``model='garch'`` : ``(omega, alpha, beta)``
        - ``model='gjr'``   : ``(omega, alpha, gamma, beta)``
        - ``model='egarch'``: ``(omega, alpha, gamma, beta)``

        With ``dist='t'`` the degrees of freedom ``nu`` (``> 2``) is appended
        as the final element.
    y : np.ndarray[np.float64, ndim=1]
        Mean-zero innovation (return) series.
    model : {'garch', 'gjr', 'egarch'}, optional
        Conditional-variance specification. ``'garch'`` is the vanilla
        GARCH(1, 1); ``'gjr'`` adds a leverage term
        ``gamma * 1[y_{t-1} < 0] * y_{t-1}^2``; ``'egarch'`` models the
        log-variance. Default is ``'garch'``.
    dist : {'normal', 't'}, optional
        Innovation density: Gaussian, or standardized (unit-variance)
        Student-t with ``nu > 2`` degrees of freedom. Default is ``'normal'``.

    Returns
    -------
    float
        Total log-likelihood (higher is better), or ``-np.inf`` when the
        parameters leave the admissible region (see Notes).

    Notes
    -----
    Admissible regions (outside them the return is ``-np.inf``):

    - ``garch`` : ``omega > 0``, ``alpha >= 0``, ``beta >= 0``,
      ``alpha + beta < 1`` (stationarity).
    - ``gjr`` : ``omega > 0``, ``alpha >= 0``, ``beta >= 0``,
      ``alpha + gamma >= 0`` (variance non-negativity),
      ``alpha + beta + gamma / 2 < 1`` (stationarity, symmetric innovations).
    - ``egarch`` : ``|beta| < 1`` only (the log-variance form needs no
      non-negativity constraint).
    - ``dist='t'`` : ``nu > 2`` (finite variance).

    The conditional variance is initialised at ``sigma_0^2 = omega`` (GARCH,
    GJR) or ``ln sigma_0^2 = omega`` (EGARCH): only the constant survives at
    ``t = 0``, matching the :func:`_arma_garch` convention. The EGARCH
    ``E|z|`` term is ``sqrt(2 / pi)`` for the normal and
    ``2 * sqrt(nu - 2) * Gamma((nu + 1) / 2) / ((nu - 1) * sqrt(pi)
    * Gamma(nu / 2))`` for the standardized Student-t.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal(500)
    >>> ll = loglik_garch([0.1, 0.05, 0.9], y, model='garch', dist='normal')
    >>> bool(ll < 0.0)
    True
    >>> loglik_garch([-1.0, 0.05, 0.9], y, model='garch')  # omega <= 0
    -inf

    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    model = model.lower()
    dist = dist.lower()

    # Split off the Student-t degrees of freedom, if any.
    if dist == 't':
        nu = float(params[-1])
        core = params[:-1]
        if nu <= 2.0:
            return -np.inf
    elif dist == 'normal':
        nu = 0.0
        core = params
    else:
        raise ValueError(f"Unknown dist: {dist!r}")

    # Parse per model, validate the admissible region, run the filter.
    if model == 'garch':
        omega, alpha, beta = core[0], core[1], core[2]
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
            return -np.inf
        h = _gjr_garch(y, omega, alpha, 0.0, beta)

    elif model == 'gjr':
        omega, alpha, gamma, beta = core[0], core[1], core[2], core[3]
        if (omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + gamma < 0.0
                or alpha + beta + 0.5 * gamma >= 1.0):
            return -np.inf
        h = _gjr_garch(y, omega, alpha, gamma, beta)

    elif model == 'egarch':
        omega, alpha, gamma, beta = core[0], core[1], core[2], core[3]
        if abs(beta) >= 1.0:
            return -np.inf
        if dist == 't':
            mean_abs_z = _mean_abs_standardized_t(nu)
        else:
            mean_abs_z = _SQRT_2_OVER_PI
        h = _egarch(y, omega, alpha, gamma, beta, mean_abs_z)

    else:
        raise ValueError(f"Unknown model: {model!r}")

    # Innovation log-density.
    if dist == 't':
        ll = _loglik_t(y, h, nu)
    else:
        ll = _loglik_normal(y, h)

    if not np.isfinite(ll):
        return -np.inf

    return float(ll)
