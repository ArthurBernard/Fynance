#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin
cimport numpy as np
from libc.math cimport sqrt, pi, log
#from libcpp cimport bool as bool_t
from cpython cimport bool

from fynance.models.econometric_models import model_ARMA_GARCH_cython, get_parameters


#=============================================================================#
#                                 ESTIMATION                                  #
#=============================================================================#


cpdef tuple estimation(
        np.ndarray[np.float64_t, ndim=1] y,
        np.ndarray[np.float64_t, ndim=1] x0,
        int p, int q, int Q, int P, bool cons, 
    ):
    """ 
    NOT YET WORKING !
    Estimator 
    """
    cdef np.ndarray[np.float64_t, ndim=1] params=np.zeros([p+q+Q+P+2], dtype=np.float64)#= fmin(
    #    target_function, x0, args=(p, q, Q, P, cons, y), disp=0
    #)
    cdef np.ndarray[np.float64_t, ndim=1] phi, theta, alpha, beta, u, h
    cdef np.float64_t c, omega, L
    
    phi, theta, alpha, beta, c, omega = get_parameters(
        params, p, q, Q, P, cons
    )
    u, h = model_ARMA_GARCH_cython(
        y, phi, theta, alpha, beta, c, omega, p, q, Q, P
    )
    L = loglikelihood(u, h)
    return u, h, phi, theta, alpha, beta, c, omega, L


cdef np.float64_t target_function(
        np.ndarray[np.float64_t, ndim=1] params,
        int p, int q, int Q, int P, bool cons,
        np.ndarray[np.float64_t, ndim=1] y,
    ):
    """ Target function """
    cdef np.ndarray[np.float64_t, ndim=1] phi, theta, alpha, beta, u, h
    cdef np.float64_t c, omega, L

    phi, theta, alpha, beta, c, omega = get_parameters(
    params, p, q, Q, P, cons
    )
    u, h = model_ARMA_GARCH_cython(
        y, phi, theta, alpha, beta, c, omega, p, q, Q, P
    )
    L = loglikelihood(u, h)
    return L
        

#=============================================================================#
#                                DISTRIBUTION                                 #
#=============================================================================#


cdef np.float64_t loglikelihood(
        np.ndarray[np.float64_t, ndim=1] u,
        np.ndarray[np.float64_t, ndim=1] h
    ):
    """ Normal log-likelihood function """
    cdef np.float64_t L
    cdef np.float64_t l_sq_pi = log(sqrt(pi))
    
    L = 0.5 - l_sq_pi - sum(h) - 0.5 * u / h
    return L