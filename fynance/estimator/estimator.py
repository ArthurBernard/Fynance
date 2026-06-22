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
    """ Objective (cost) to minimise when fitting an ARMA/GARCH model.

    Splits the flat ``params`` vector with :func:`get_parameters`, runs the
    selected model recursion over ``y`` to obtain the residuals ``u`` (and the
    conditional volatility ``h`` for GARCH models), then returns the negative
    Gaussian log-likelihood (see :func:`loglikelihood`). Smaller is better, so
    the value can be handed straight to a minimiser.

    Parameters
    ----------
    params : np.ndarray[np.float64, ndim=1]
        Flat parameter vector laid out as expected by :func:`get_parameters`
        for the given ``p, q, Q, P, cons`` configuration.
    y : np.ndarray[np.float64, ndim=1]
        Time series to fit.
    p, q : int, optional
        Orders of the AR and MA parts of the ARMA mean equation. Default is 0.
    Q, P : int, optional
        Orders of the ARCH and GARCH parts of the conditional variance.
        Default is 0.
    cons : bool, optional
        Whether ``params`` includes a leading constant term. Default is True.
    model : {'arch', 'garch', 'arma'}, optional
        Model family driving the residual recursion. ``'arch'`` and
        ``'garch'`` both use the ARMA-GARCH recursion; ``'arma'`` uses the ARMA
        recursion with unit conditional volatility. Default is ``'arch'``.

    Returns
    -------
    np.float64
        Negative Gaussian log-likelihood of the residuals (cost to minimise).

    Raises
    ------
    ValueError
        If ``model`` is not one of ``'arch'``, ``'garch'`` or ``'arma'``.

    See Also
    --------
    loglikelihood, get_parameters

    """
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
    r""" Negative Gaussian log-likelihood (a cost to minimise).

    Despite its name, this function returns the *negative* of the Normal
    log-likelihood of the residuals, i.e. a cost suitable for direct
    minimisation by an optimiser (smaller is better). The likelihood itself is
    the opposite of the returned value.

    .. math::

        -\ln \mathcal{L} = \frac{1}{2}\left(T \ln(2\pi)
        + \sum_t \ln(h_t^2) + \sum_t \left(\frac{u_t}{h_t}\right)^2\right)

    The input ``h`` is left unchanged: a working copy is used and its zero
    entries are floored to ``1e-8`` to avoid division by zero.

    Parameters
    ----------
    u : np.ndarray[dtype=np.float64, ndim=1]
        Standardized residuals series.
    h : np.ndarray[dtype=np.float64, ndim=1]
        Conditional standard deviation series of residuals. Not modified in
        place.

    Returns
    -------
    np.float64
        Negative Gaussian log-likelihood of the residuals (cost to minimise).

    """
    l_sq_pi = np.log(2 * np.pi)
    T = h.size
    h = np.array(h, dtype=np.float64, copy=True)
    h[h == 0] = 1e-8
    L = T * l_sq_pi + np.sum(np.log(np.square(h))) + np.sum(np.square(u / h))

    return 0.5 * L
