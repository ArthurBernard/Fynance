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

from fynance.estimator.estimator_cy import loglikelihood_cy
from fynance.models.econometric_models import get_parameters

# Internal packages
from fynance.models.econometric_models_cy import ARMA_cy, ARMA_GARCH_cy

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

       For ARMA/GARCH parameter estimation use the Cython-backed path exposed
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
        "parameter estimation (the Cython-backed, authoritative path)."
    )


def target_function(params, y, p=0, q=0, Q=0, P=0, cons=True, model='arch'):
    """ Target function """
    phi, theta, alpha, beta, c, omega = get_parameters(
        params, p, q, Q, P, cons
    )

    if model.lower() == 'arch' or model.lower() == 'garch':
        u, h = ARMA_GARCH_cy(
            y, phi, theta, alpha, beta, c, omega, p, q, Q, P
        )

    elif model.lower() == 'arma':
        u = ARMA_cy(y, phi, theta, c, p, q)
        h = np.ones([u.size], dtype=np.float64)

    else:
        raise ValueError(f"Unknown model: {model!r}")

    L = loglikelihood_cy(u, h)

    return L


# =========================================================================== #
#                                DISTRIBUTION                                 #
# =========================================================================== #


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
