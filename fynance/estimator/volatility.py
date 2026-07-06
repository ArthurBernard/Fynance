#!/usr/bin/env python3
# coding: utf-8

r""" Maximum-likelihood driver for GARCH-family volatility models.

Exposes :func:`fit_volatility` — a :mod:`scipy.optimize`-based maximum-likelihood
fit of a GARCH / GJR-GARCH / EGARCH(1, 1) model on a return series — and its
result container :class:`VolatilityResult`, which carries the fitted parameters,
standard errors, information criteria, the in-sample conditional volatility and
standardized residuals, plus closed-form / Monte-Carlo variance forecasting and
model simulation.

The single authoritative conditional-variance recursions and log-likelihood live
in :mod:`fynance.models.econometric_models` (the Numba kernels ``_gjr_garch`` /
``_egarch`` and :func:`~fynance.models.econometric_models.loglik_garch`). This
module only wires the optimiser, the standard errors and the forecasting /
simulation on top; it does **not** re-derive the recursions or the parameter
layout.

Parameter layout
----------------
The flat parameter vector handed to :func:`~fynance.models.econometric_models.\
loglik_garch` is (variance parameters first, ``nu`` appended last for ``'t'``):

- ``model='garch'`` : ``(omega, alpha, beta)``
- ``model='gjr'``   : ``(omega, alpha, gamma, beta)``
- ``model='egarch'``: ``(omega, alpha, gamma, beta)``

with ``nu`` (``> 2``) appended when ``dist='t'``.
"""

from __future__ import annotations

# Built-in packages
import math
from dataclasses import dataclass

# Third-party packages
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

# Local packages
from fynance.models.econometric_models import (
    _egarch,
    _gjr_garch,
    _mean_abs_standardized_t,
    loglik_garch,
)

__all__ = ['VolatilityResult', 'fit_volatility']

# E|z| for a standard-normal innovation (EGARCH asymmetry-centering constant).
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

# Stationarity is enforced strictly inside the open region via this slack.
_EPS_STAT = 1e-6


# =========================================================================== #
#                             PARAMETER PLUMBING                              #
# =========================================================================== #


def _param_names(model: str, dist: str) -> list[str]:
    """ Ordered parameter names matching the flat kernel vector. """
    if model == 'garch':
        names = ['omega', 'alpha', 'beta']
    elif model in ('gjr', 'egarch'):
        names = ['omega', 'alpha', 'gamma', 'beta']
    else:
        raise ValueError(f"Unknown model: {model!r}")

    if dist == 't':
        names = names + ['nu']
    elif dist != 'normal':
        raise ValueError(f"Unknown dist: {dist!r}")

    return names


def _unpack(params: dict[str, float], model: str) -> tuple[float, float, float,
                                                           float]:
    """ Return ``(omega, alpha, gamma, beta)`` from a parameter mapping. """
    omega = float(params['omega'])
    alpha = float(params['alpha'])
    beta = float(params['beta'])
    gamma = float(params.get('gamma', 0.0))

    return omega, alpha, gamma, beta


def _filter_vol(
    y: NDArray[np.float64], model: str, dist: str, params: dict[str, float],
) -> NDArray[np.float64]:
    """ Forward-filter the conditional volatility with the right kernel.

    Runs the authoritative Numba recursion (``_gjr_garch`` for garch / gjr,
    ``_egarch`` for egarch) over the mean-zero series ``y`` and returns the
    conditional standard deviation :math:`\\sigma_t`.

    Parameters
    ----------
    y : numpy.ndarray
        Mean-zero innovation (return) series.
    model : {'garch', 'gjr', 'egarch'}
        Conditional-variance specification.
    dist : {'normal', 't'}
        Innovation density (only used by egarch, to center ``|z|``).
    params : dict of str to float
        Fitted parameters (``omega, alpha, [gamma,] beta, [nu]``).

    Returns
    -------
    numpy.ndarray
        Conditional standard deviation :math:`\\sigma_t`, same length as ``y``.

    """
    omega, alpha, gamma, beta = _unpack(params, model)

    if model == 'garch':
        return np.asarray(_gjr_garch(y, omega, alpha, 0.0, beta))

    if model == 'gjr':
        return np.asarray(_gjr_garch(y, omega, alpha, gamma, beta))

    if model == 'egarch':
        if dist == 't':
            mean_abs_z = _mean_abs_standardized_t(float(params['nu']))
        else:
            mean_abs_z = _SQRT_2_OVER_PI

        return np.asarray(_egarch(y, omega, alpha, gamma, beta, mean_abs_z))

    raise ValueError(f"Unknown model: {model!r}")


def _draw(rng: np.random.Generator, dist: str, nu: float,
          size: int) -> NDArray[np.float64]:
    """ Draw ``size`` unit-variance innovations from ``dist``. """
    if dist == 'normal':
        return rng.standard_normal(size)

    if dist == 't':
        # standard_t has variance nu / (nu - 2); rescale to unit variance.
        return rng.standard_t(nu, size) * math.sqrt((nu - 2.0) / nu)

    raise ValueError(f"Unknown dist: {dist!r}")


# =========================================================================== #
#                             OPTIMISER SETUP                                 #
# =========================================================================== #


def _neg_loglik(params: NDArray[np.float64], y: NDArray[np.float64],
                model: str, dist: str) -> float:
    """ Negative log-likelihood (cost to minimise); +1e10 on inadmissible.

    The egarch log-variance recursion can underflow to a zero conditional
    volatility for extreme parameters, which the Numba kernel surfaces as a
    ``ZeroDivisionError`` (its ``error_model`` mirrors Python); such points are
    treated as inadmissible (the optimiser is repelled with the same penalty).
    """
    try:
        ll = loglik_garch(params, y, model, dist)
    except (ZeroDivisionError, FloatingPointError):
        return 1e10

    if not np.isfinite(ll):
        return 1e10

    return -ll


def _x0_heuristic(y: NDArray[np.float64], model: str, dist: str,
                  x0: ArrayLike | None) -> NDArray[np.float64]:
    r""" Initial parameter vector via variance targeting.

    With ``alpha = 0.05``, ``beta = 0.90``, ``gamma = 0.05`` and (for ``'t'``)
    ``nu = 8``, variance targeting sets ``omega`` so the model's unconditional
    variance matches the sample variance: for garch / gjr this is
    ``omega = var * (1 - alpha - beta)``. For egarch ``omega`` is the
    *log-variance* intercept, so the same targeting is applied in log space,
    ``omega = ln(var) * (1 - beta)`` (its unconditional log-variance is
    ``omega / (1 - beta)``). An explicit ``x0`` overrides the heuristic.
    """
    if x0 is not None:
        return np.asarray(x0, dtype=np.float64).reshape(-1)

    var = float(np.var(y))
    if var <= 0.0:
        var = 1e-8

    alpha, beta, gamma, nu = 0.05, 0.90, 0.05, 8.0

    if model == 'garch':
        core = [var * (1.0 - alpha - beta), alpha, beta]
    elif model == 'gjr':
        core = [var * (1.0 - alpha - beta), alpha, gamma, beta]
    elif model == 'egarch':
        core = [math.log(var) * (1.0 - beta), alpha, gamma, beta]
    else:
        raise ValueError(f"Unknown model: {model!r}")

    if dist == 't':
        core = core + [nu]

    return np.asarray(core, dtype=np.float64)


def _bounds_and_constraints(
    model: str, dist: str,
) -> tuple[list[tuple[float | None, float | None]], list[dict]]:
    """ Box bounds and stationarity constraints for the SLSQP optimiser. """
    bounds: list[tuple[float | None, float | None]]
    if model == 'garch':
        bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]
        # alpha + beta < 1.
        cons = [{'type': 'ineq',
                 'fun': lambda p: 1.0 - _EPS_STAT - p[1] - p[2]}]
    elif model == 'gjr':
        bounds = [(1e-12, None), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]
        # alpha + gamma >= 0 (variance non-negativity) and
        # alpha + beta + gamma / 2 < 1 (stationarity, symmetric innovations).
        cons = [
            {'type': 'ineq', 'fun': lambda p: p[1] + p[2]},
            {'type': 'ineq',
             'fun': lambda p: 1.0 - _EPS_STAT - p[1] - p[3] - 0.5 * p[2]},
        ]
    elif model == 'egarch':
        # Log-variance form: only |beta| < 1 is required (box bound below).
        bounds = [(None, None), (None, None), (None, None), (-0.999, 0.999)]
        cons = []
    else:
        raise ValueError(f"Unknown model: {model!r}")

    if dist == 't':
        bounds = bounds + [(2.05, None)]

    return bounds, cons


def _numerical_hessian(x: NDArray[np.float64], y: NDArray[np.float64],
                       model: str, dist: str) -> NDArray[np.float64]:
    """ Central finite-difference Hessian of the negative log-likelihood. """
    n = x.size
    step = 1e-4 * np.maximum(np.abs(x), 1e-3)
    f0 = _neg_loglik(x, y, model, dist)
    hess = np.empty((n, n))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        for j in range(i, n):
            if i == j:
                fp = _neg_loglik(x + ei, y, model, dist)
                fm = _neg_loglik(x - ei, y, model, dist)
                hess[i, i] = (fp - 2.0 * f0 + fm) / (step[i] ** 2)
            else:
                ej = np.zeros(n)
                ej[j] = step[j]
                fpp = _neg_loglik(x + ei + ej, y, model, dist)
                fpm = _neg_loglik(x + ei - ej, y, model, dist)
                fmp = _neg_loglik(x - ei + ej, y, model, dist)
                fmm = _neg_loglik(x - ei - ej, y, model, dist)
                val = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])
                hess[i, j] = val
                hess[j, i] = val

    return hess


def _std_errors(x: NDArray[np.float64], y: NDArray[np.float64],
                model: str, dist: str) -> dict[str, float]:
    """ Standard errors from the inverse observed-information (Hessian).

    Inverts the numerical Hessian of the negative log-likelihood at the
    optimum. A non-positive-definite Hessian (or a singular inversion) yields
    ``NaN`` standard errors rather than raising.
    """
    names = _param_names(model, dist)
    se = np.full(x.size, np.nan)
    hess = _numerical_hessian(x, y, model, dist)

    if np.all(np.isfinite(hess)):
        try:
            # A max of the log-likelihood has a positive-definite Hessian of
            # the *negative* log-likelihood; cholesky guards that.
            np.linalg.cholesky(hess)
            diag = np.diag(np.linalg.inv(hess))
            se = np.where(diag > 0.0, np.sqrt(diag), np.nan)
        except np.linalg.LinAlgError:
            pass

    return dict(zip(names, se))


# =========================================================================== #
#                                 RESULT                                      #
# =========================================================================== #


@dataclass
class VolatilityResult:
    r""" Fitted GARCH-family volatility model (see :func:`fit_volatility`).

    Attributes
    ----------
    params : dict of str to float
        Fitted parameters, keyed by name (``omega, alpha, [gamma,] beta,
        [nu]``).
    std_errors : dict of str to float
        Standard error of each parameter (``NaN`` when the observed-information
        Hessian is not positive definite).
    loglik : float
        Maximised log-likelihood.
    aic, bic : float
        Akaike / Bayesian information criteria, ``2k - 2ll`` and
        ``k ln(n) - 2ll`` with ``k`` parameters and ``n`` observations.
    conditional_vol : numpy.ndarray
        In-sample conditional standard deviation :math:`\sigma_t`, shape
        ``(n_obs,)``.
    std_residuals : numpy.ndarray
        Standardized residuals :math:`y_t / \sigma_t`, shape ``(n_obs,)``.
    model : {'garch', 'gjr', 'egarch'}
        Conditional-variance specification.
    dist : {'normal', 't'}
        Innovation density.
    n_obs : int
        Number of observations the model was fit on.

    """

    params: dict[str, float]
    std_errors: dict[str, float]
    loglik: float
    aic: float
    bic: float
    conditional_vol: NDArray[np.float64]
    std_residuals: NDArray[np.float64]
    model: str
    dist: str
    n_obs: int

    def forecast(self, h: int = 1, n_sims: int = 10_000,
                 seed: int = 0) -> NDArray[np.float64]:
        r""" Multi-step conditional-variance forecast.

        Returns :math:`\mathbb E[\sigma_{T+k}^2 \mid \mathcal F_{T-1}]` for
        ``k = 1, \dots, h`` (``T = n_obs``). The one-step value is exactly the
        filter's next-step variance.

        For **garch** / **gjr** the forecast is the closed-form recursion
        toward the unconditional variance: after the exact one-step step, the
        leverage indicator is replaced by its expectation
        :math:`\mathbb E[\mathbf 1[\varepsilon < 0]] = 1/2` (symmetric
        innovations), giving persistence :math:`\alpha + \beta + \gamma / 2`.
        For **egarch** (log-variance form) no closed form exists, so the
        forecast is a seeded Monte-Carlo average over ``n_sims`` simulated
        innovation paths.

        Parameters
        ----------
        h : int, optional
            Forecast horizon (number of steps). Default 1.
        n_sims : int, optional
            Number of Monte-Carlo paths (egarch only). Default 10000.
        seed : int, optional
            Seed for the Monte-Carlo draws (egarch only). Default 0.

        Returns
        -------
        numpy.ndarray
            Variance forecasts, shape ``(h,)``.

        """
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}")

        omega, alpha, gamma, beta = _unpack(self.params, self.model)
        sig_last = float(self.conditional_vol[-1])
        eps_last = float(self.std_residuals[-1]) * sig_last
        out = np.empty(h, dtype=np.float64)

        if self.model in ('garch', 'gjr'):
            ind = 1.0 if eps_last < 0.0 else 0.0
            out[0] = (omega + (alpha + gamma * ind) * eps_last ** 2
                      + beta * sig_last ** 2)
            persist = alpha + beta + 0.5 * gamma
            for k in range(1, h):
                out[k] = omega + persist * out[k - 1]

            return out

        # egarch: Monte-Carlo forward simulation of the log-variance.
        nu = float(self.params.get('nu', 0.0))
        if self.dist == 't':
            mean_abs_z = _mean_abs_standardized_t(nu)
        else:
            mean_abs_z = _SQRT_2_OVER_PI

        z_last = float(self.std_residuals[-1])
        log_var_next = (omega + beta * math.log(sig_last ** 2)
                        + alpha * (abs(z_last) - mean_abs_z) + gamma * z_last)
        out[0] = math.exp(log_var_next)

        if h > 1:
            rng = np.random.default_rng(seed)
            log_var = np.full(n_sims, log_var_next)
            for k in range(1, h):
                z = _draw(rng, self.dist, nu, n_sims)
                log_var = (omega + beta * log_var
                           + alpha * (np.abs(z) - mean_abs_z) + gamma * z)
                out[k] = float(np.mean(np.exp(log_var)))

        return out

    def simulate(self, T: int, seed: int = 0) -> NDArray[np.float64]:
        r""" Simulate a return path from the fitted parameters and innovation.

        Runs the fitted model's conditional-variance recursion forward,
        drawing unit-variance innovations from the fitted ``dist`` (Gaussian,
        or standardized Student-t with ``nu`` degrees of freedom), and returns
        the mean-zero simulated returns :math:`y_t = \sigma_t z_t`.

        Parameters
        ----------
        T : int
            Length of the simulated path.
        seed : int, optional
            Seed for the innovation draws. Default 0.

        Returns
        -------
        numpy.ndarray
            Simulated returns, shape ``(T,)``.

        """
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")

        omega, alpha, gamma, beta = _unpack(self.params, self.model)
        nu = float(self.params.get('nu', 0.0))
        rng = np.random.default_rng(seed)
        z = _draw(rng, self.dist, nu, T)
        y = np.empty(T, dtype=np.float64)

        if self.model in ('garch', 'gjr'):
            var = omega  # sigma_0^2 = omega, matching the kernel convention.
            y[0] = math.sqrt(var) * z[0]
            for t in range(1, T):
                ind = 1.0 if y[t - 1] < 0.0 else 0.0
                var = (omega + (alpha + gamma * ind) * y[t - 1] ** 2
                       + beta * var)
                y[t] = math.sqrt(var) * z[t]

            return y

        # egarch.
        if self.dist == 't':
            mean_abs_z = _mean_abs_standardized_t(nu)
        else:
            mean_abs_z = _SQRT_2_OVER_PI

        log_var = omega  # ln sigma_0^2 = omega, matching the kernel.
        y[0] = math.exp(0.5 * log_var) * z[0]
        for t in range(1, T):
            log_var = (omega + beta * log_var
                       + alpha * (abs(z[t - 1]) - mean_abs_z) + gamma * z[t - 1])
            y[t] = math.exp(0.5 * log_var) * z[t]

        return y


# =========================================================================== #
#                                   DRIVER                                    #
# =========================================================================== #


def fit_volatility(
    y: ArrayLike,
    model: str = 'garch',
    dist: str = 'normal',
    x0: ArrayLike | None = None,
) -> VolatilityResult:
    r""" Maximum-likelihood fit of a GARCH-family volatility model.

    Demeans ``y`` (the sample mean is removed first) and maximises the
    GARCH-family log-likelihood
    (:func:`~fynance.models.econometric_models.loglik_garch`) with
    :func:`scipy.optimize.minimize` (``method='SLSQP'``), under box bounds and
    stationarity constraints. The starting point is the variance-targeting
    heuristic (see Notes); an explicit ``x0`` overrides it. Standard errors come
    from the inverse numerical Hessian of the negative log-likelihood.

    Parameters
    ----------
    y : array-like
        One-dimensional return series (demeaned internally).
    model : {'garch', 'gjr', 'egarch'}, optional
        Conditional-variance specification. ``'garch'`` is vanilla
        GARCH(1, 1); ``'gjr'`` adds a leverage term; ``'egarch'`` models the
        log-variance. Default is ``'garch'``.
    dist : {'normal', 't'}, optional
        Innovation density: Gaussian or standardized Student-t (``nu > 2``).
        Default is ``'normal'``.
    x0 : array-like, optional
        Explicit starting parameter vector (in the layout described in the
        module docstring). Overrides the variance-targeting heuristic.

    Returns
    -------
    VolatilityResult
        Fitted parameters, standard errors, information criteria, in-sample
        conditional volatility, standardized residuals, and forecasting /
        simulation methods.

    Notes
    -----
    The starting point uses variance targeting with ``alpha = 0.05``,
    ``beta = 0.90``, ``gamma = 0.05`` and (for ``'t'``) ``nu = 8``, and sets
    ``omega`` so the model's unconditional variance matches the sample
    variance: ``omega = var * (1 - alpha - beta)`` for garch / gjr and, in log
    space, ``omega = ln(var) * (1 - beta)`` for egarch.

    Information criteria are ``AIC = 2k - 2ll`` and ``BIC = k ln(n) - 2ll``
    (``k`` parameters, ``n`` observations, ``ll`` the maximised
    log-likelihood).

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.estimator import fit_volatility
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal(300) * 0.02
    >>> res = fit_volatility(y, model='garch', dist='normal')
    >>> sorted(res.params)
    ['alpha', 'beta', 'omega']
    >>> res.conditional_vol.shape
    (300,)
    >>> bool(np.isfinite(res.aic) and np.isfinite(res.bic))
    True
    >>> bool(np.all(res.forecast(5) > 0.0))
    True

    """
    model = model.lower()
    dist = dist.lower()
    names = _param_names(model, dist)  # validates model / dist.

    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    arr = arr - np.mean(arr)
    n = arr.size

    x_start = _x0_heuristic(arr, model, dist, x0)
    bounds, cons = _bounds_and_constraints(model, dist)

    res = minimize(
        _neg_loglik,
        x_start,
        args=(arr, model, dist),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-9},
    )

    x = np.asarray(res.x, dtype=np.float64)
    params = dict(zip(names, (float(v) for v in x)))
    ll = float(loglik_garch(x, arr, model, dist))
    k = x.size
    aic = 2.0 * k - 2.0 * ll
    bic = k * math.log(n) - 2.0 * ll

    cond_vol = _filter_vol(arr, model, dist, params)
    std_resid = arr / cond_vol

    return VolatilityResult(
        params=params,
        std_errors=_std_errors(x, arr, model, dist),
        loglik=ll,
        aic=aic,
        bic=bic,
        conditional_vol=cond_vol,
        std_residuals=std_resid,
        model=model,
        dist=dist,
        n_obs=n,
    )
