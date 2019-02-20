#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in packages
from libc.math cimport sqrt, pi, log
from cpython cimport bool

# External packages
import numpy as np
from scipy.optimize import fmin
cimport numpy as np

# Internal packages
from fynance.models.econometric_models_cy import ARMA_cy, ARMA_GARCH_cy
from fynance.models.econometric_models_cy import get_parameters_cy

__all__ = ['estimation_cy', 'target_function_cy', 'loglikelihood_cy']

#=============================================================================#
#                                 ESTIMATION                                  #
#=============================================================================#


cpdef tuple estimation_cy(
        np.ndarray[np.float64_t, ndim=1] y,
        np.ndarray[np.float64_t, ndim=1] x0,
        int p=0, int q=0, int Q=0, int P=0, 
        bool cons=True,
        str model='arch' 
    ):
    """ 
    NOT YET WORKING !
    Estimator 
    """
    cdef np.ndarray[np.float64_t, ndim=1] params=np.zeros([p+q+Q+P+2], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] phi, theta, alpha, beta, u, h
    cdef np.float64_t c, omega, L
    
    params = fmin(target_function_cy, x0, args=(y, p, q, Q, P, cons), disp=0)
    # NEED TO FIND AN OPTIMIZER (in C or C++) #
    phi, theta, alpha, beta, c, omega = get_parameters_cy(
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
        print('Unknow model.')
        raise ValueError
    L = loglikelihood_cy(u, h)
    return u, h, phi, theta, alpha, beta, c, omega, L


cpdef np.float64_t target_function_cy(
        np.ndarray[np.float64_t, ndim=1] params,
        np.ndarray[np.float64_t, ndim=1] y,
        int p=0, int q=0, int Q=0, int P=0, 
        bool cons=True,
        str model='arch'
    ):
    """ Target function """
    cdef np.ndarray[np.float64_t, ndim=1] phi, theta, alpha, beta, u, h
    cdef np.float64_t c, omega, L

    phi, theta, alpha, beta, c, omega = get_parameters_cy(
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
        print('Unknow model.')
        raise ValueError
    L = loglikelihood_cy(u, h)
    return L
        

#=============================================================================#
#                                DISTRIBUTION                                 #
#=============================================================================#


cpdef np.float64_t loglikelihood_cy(
        np.ndarray[np.float64_t, ndim=1] u,
        np.ndarray[np.float64_t, ndim=1] h
    ):
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
    cdef np.float64_t L, l_sq_pi = log(2 * pi)
    cdef int T = u.size
    h = np.square(h) + 1e-8
    
    L = <double>T * l_sq_pi + np.sum(np.log(h)) + np.sum(np.square(u) / h)
    return 0.5 * L