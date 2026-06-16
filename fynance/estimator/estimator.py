#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-20 10:40:45
# @Last modified by: ArthurBernard
# @Last modified time: 2019-05-23 18:09:58

""" Tools to estimate models. """

# Built-in packages

# External packages
import numpy as np

from fynance.models.econometric_models import ARMA, ARMA_GARCH, get_parameters

__all__ = ['estimation', 'target_function', 'loglikelihood']

# =========================================================================== #
#                                 ESTIMATION                                  #
# =========================================================================== #


def estimation(y, x0, p=0, q=0, Q=0, P=0, cons=True, model='arch'):
    """ Estimate ARMA/GARCH parameters by maximum likelihood.

    .. warning::

       **Experimental — not implemented.** This pure-Python optimisation
       driver never reached a working optimiser. It is kept as a placeholder
       only; calling it raises :class:`NotImplementedError`.

       For ARMA/GARCH parameter estimation use the Numba-backed path exposed
       through :func:`fynance.models.econometric_models.get_parameters`, which
       is the authoritative implementation (the Python layer must not duplicate
       it — see the estimator stability policy).

    Raises
    ------
    NotImplementedError
        Always — see the warning above.

    """
    raise NotImplementedError(
        "fynance.estimator.estimation is experimental and not implemented; "
        "use fynance.models.econometric_models.get_parameters for ARMA/GARCH "
        "parameter estimation (the Numba-backed, authoritative path)."
    )


def target_function(params, y, p=0, q=0, Q=0, P=0, cons=True, model='arch'):
    """ Target function """
    phi, theta, alpha, beta, c, omega = get_parameters(
        params, p, q, Q, P, cons
    )

    if model.lower() == 'arch' or model.lower() == 'garch':
        u, h = ARMA_GARCH(y, phi, theta, alpha, beta, c, omega, p, q, Q, P)

    elif model.lower() == 'arma':
        u = ARMA(y, phi, theta, c, p, q)
        h = np.ones([u.size], dtype=np.float64)

    else:
        raise ValueError(f"Unknown model: {model!r}")

    return _loglikelihood(u, h)


# =========================================================================== #
#                                DISTRIBUTION                                 #
# =========================================================================== #


def _loglikelihood(u, h):
    """ Normal log-likelihood.

    Adds a 1e-8 floor to every conditional variance term (not only zeros), as
    the original kernel did; used internally by :func:`target_function`.
    """
    h2 = np.square(h) + 1e-8
    L = u.size * np.log(2 * np.pi) + np.sum(np.log(h2)) + np.sum(np.square(u) / h2)

    return 0.5 * L


def loglikelihood(u, h):
    """ Normal log-likelihood function.

    Parameters
    ----------
    u : np.ndarray[dtype=np.float64, ndim=1]
        Standardized residuals series.
    h : np.ndarray[dtype=np.float64, ndim=1]
        Conditional standard deviation series of residuals.

    Returns
    -------
    np.float64
        Normal log likelihood of residuals.

    """
    l_sq_pi = np.log(2 * np.pi)
    T = h.size
    h[h == 0] = 1e-8
    L = T * l_sq_pi + np.sum(np.log(np.square(h))) + np.sum(np.square(u / h))

    return 0.5 * L
