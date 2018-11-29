#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin

from fynance.models.econometric_models_cy import *
from .estimator_cy import *

__all__ = ['estimation', 'target_function', 'loglikelihood']

#=============================================================================#
#                                 ESTIMATION                                  #
#=============================================================================#


def estimation(y, x0, p=0, q=0, Q=0, P=0, cons=True):
    """ 
    NOT YET WORKING !
    Estimator 
    """
    params = fmin(target_function, x0, args=(y, p, q, Q, P, cons), disp=0)
    # NEED TO FIND AN OPTIMIZER #
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
    L = loglikelihood(u, h)
    return u, h, phi, theta, alpha, beta, c, omega, L


def target_function(params, y, p=0, q=0, Q=0, P=0, cons=True, model='arch'):
    """ Target function """
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
    L = loglikelihood(u, h)
    return L
        

#=============================================================================#
#                                DISTRIBUTION                                 #
#=============================================================================#


def loglikelihood(u, h):
    """ Normal log-likelihood function """
    l_sq_pi = log(sqrt(pi))
    L = - 0.5 - l_sq_pi - np.sum(h) - 0.5 * np.sum(u / (h + 1e-8))
    return - L