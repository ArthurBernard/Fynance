#!/usr/bin/env python3
# coding: utf-8

""" GARCH(1,1) conditional volatility as a causal feature.

Exposes :func:`garch_volatility` — a strictly **causal** conditional-volatility
series derived from a GARCH(1,1) fit. The single authoritative ARMA/GARCH
implementation lives in :mod:`fynance.models.econometric_models` (the Numba
recursion) and :mod:`fynance.estimator` (the likelihood); this module only adds
the thin maximum-likelihood fit + forward-filter scheme, it does **not**
re-derive the recursion or the parameter layout.

Causality
---------
Two things could leak the future into the feature:

- the **parameters** — fit once on a training prefix (expanding-fit), optionally
  refit on the expanding window every ``refit`` steps; never on the whole series;
- the **filtered volatility** — :math:`\\sigma_t` in GARCH is
  :math:`\\mathcal F_{t-1}`-measurable (it depends on
  :math:`u_{t-1}, \\sigma_{t-1}`, not on :math:`u_t`), so running the recursion
  forward is causal given fixed parameters.

The first ``min_train`` points (the warmup whose parameters would be in-sample)
are returned as ``NaN``.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

# Local packages
from fynance.estimator.estimator import target_function
from fynance.models.econometric_models import ARMA_GARCH, get_parameters

__all__ = ['garch_volatility']

# GARCH(1,1) model order: zero-mean-constant ARMA part, GARCH(1, 1) variance.
_P, _Q, _AR, _MA = 1, 1, 0, 0


def _fit_garch11(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Maximum-likelihood fit of a GARCH(1,1) on ``y``.

    Minimizes the negative log-likelihood (:func:`fynance.estimator.estimator.\
target_function`) over the flat parameter vector ``[c, omega, alpha, beta]``,
    under positivity and stationarity (``alpha + beta < 1``) constraints.

    Returns
    -------
    numpy.ndarray
        The fitted ``[c, omega, alpha, beta]`` vector.

    """
    var = float(np.var(y))

    if var <= 0.0:
        var = 1e-8

    x0 = np.array([float(np.mean(y)), var * 0.1, 0.05, 0.90])
    bounds = [(None, None), (1e-12, None), (0.0, 1.0), (0.0, 1.0)]
    # Stationarity: alpha + beta <= 1 - eps.
    constraints = ({
        'type': 'ineq',
        'fun': lambda p: 0.999 - p[2] - p[3],
    },)

    res = minimize(
        target_function,
        x0,
        args=(y, _AR, _MA, _Q, _P, True, 'garch'),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-8},
    )

    return np.asarray(res.x, dtype=np.float64)


def _filter_sigma(
    y: NDArray[np.float64], params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """ Forward-filter the conditional volatility on ``y`` with ``params``. """
    phi, theta, alpha, beta, c, omega = get_parameters(
        params, _AR, _MA, _Q, _P, cons=True,
    )
    _, h = ARMA_GARCH(
        y, phi, theta, alpha, beta, c, omega, _AR, _MA, _Q, _P,
    )

    return np.asarray(h, dtype=np.float64)


def garch_volatility(
    returns: ArrayLike,
    refit: int | None = None,
    min_train: int = 250,
) -> NDArray[np.float64]:
    r""" Causal GARCH(1,1) conditional-volatility feature.

    Fits a GARCH(1,1) on a training prefix and forward-filters the conditional
    volatility :math:`\sigma_t` over the whole series. The first ``min_train``
    values are ``NaN`` (the warmup whose parameters would be in-sample).

    Parameters
    ----------
    returns : array-like
        One-dimensional return series.
    refit : int, optional
        Refit the parameters on the expanding window every ``refit`` steps
        (each block uses parameters fit strictly on its own past). If ``None``
        (default), the parameters are fit once on ``returns[:min_train]``.
    min_train : int, optional
        Length of the initial training prefix; values before it are ``NaN``.
        Default 250.

    Returns
    -------
    numpy.ndarray
        Conditional volatility :math:`\sigma_t`, same length as ``returns``,
        ``NaN`` on the warmup.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(400) * 0.01
    >>> sigma = garch_volatility(r, min_train=200)
    >>> sigma.shape
    (400,)
    >>> bool(np.all(np.isnan(sigma[:200])))
    True
    >>> bool(np.all(sigma[200:] >= 0))
    True

    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = r.size

    if min_train < 2 or min_train >= n:

        raise ValueError(
            f"min_train must be in [2, len(returns)), got {min_train}"
        )

    sigma = np.full(n, np.nan, dtype=np.float64)

    # Block starts: where each parameter regime takes effect.
    if refit is None or refit <= 0:
        starts = [min_train]

    else:
        starts = list(range(min_train, n, refit))

    bounds = starts + [n]
    for k, start in enumerate(starts):
        end = bounds[k + 1]
        params = _fit_garch11(r[:start])
        # sigma_t depends on the past only, so filtering the full series with
        # these (past-fit) params and slicing [start:end] stays causal.
        h = _filter_sigma(r, params)
        sigma[start:end] = h[start:end]

    return sigma
