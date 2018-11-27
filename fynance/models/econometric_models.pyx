#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin
cimport numpy as np
from libc.math cimport sqrt, pi, log
#from libcpp cimport bool as bool_t
from cpython cimport bool


#=============================================================================#
#                             PARAMETERS FUNCTION                             #
#=============================================================================#


cpdef tuple get_parameters(
        np.ndarray[np.float64_t, ndim=1] params,
        int p, int q, int Q, int P, bool cons
    ):
    """ Get parameters for ARMA-GARCH models """
    cdef int i = 0
    cdef np.ndarray[np.float64_t, ndim=1] phi, theta, alpha, beta
    cdef np.float64_t c, omega

    if cons:
        c = params[i]
        i += 1
    if p > 0:
        phi = params[i: p+i]
        i += p
    if q > 0:
        theta = params[i: q+i]
        i += q
    if Q > 0 or P > 0:
        omega = params[i]
        i += 1
        if Q > 0:
            alpha = params[i: Q+i]
            i += Q
        if P > 0:
            beta = params[i: P+i]
            i += P
    return phi, theta, alpha, beta, c, omega


#=============================================================================#
#                                   MODELs                                    #
#=============================================================================#


cpdef np.ndarray[np.float64_t, ndim=1] model_MA_cython(
        np.ndarray[np.float64_t, ndim=1] y, 
        np.ndarray[np.float64_t, ndim=1] theta,
        np.float64_t c, int q
    ):
    """ 
    Moving Average model of order q s.t: 
    y_t = c + theta_1 * u_t-1 + ... + theta_q * u_t-q + u_t 
    
    Parameters
    ----------
    y: np.ndarray[np.float64, ndim=1]
        Time series.
    theta: np.ndarray[np.float64, ndim=1]
        Coefficients of model.
    c: np.float64
        Constant of the model.
    q: int
        Order of MA(q) model.
        
    Returns
    -------
    u: np.ndarray[ndim=1, dtype=np.float64]
        Residual of the model.
    """
    cdef np.float64_t s
    cdef int i, t, T = y.size
    cdef np.ndarray[np.float64_t, ndim=1] u = np.zeros([T], dtype=np.float64)  # Residuals
    
    for t in range(T):
        s = 0
        for i in range(min(t, q)):
            s += u[t-i-1] * theta[i]
        u[t] = y[t] - c - s
    return u


cpdef np.ndarray[np.float64_t, ndim=1] model_ARMA_cython(
        np.ndarray[np.float64_t, ndim=1] y,
        np.ndarray[np.float64_t, ndim=1] phi,
        np.ndarray[np.float64_t, ndim=1] theta,
        np.float64_t c, int p, int q
    ):
    """
    AutoRegressive Moving Average model of order q and p s.t: 
    y_t = c + phi_1 * y_t-1 + ... + phi_p * y_t-p + theta_1 * u_t-1 + ...
          + theta_q * u_t-q + u_t
    
    Parameters
    ----------
    y: np.ndarray[np.float64, ndim=1]
        Time series.
    phi: np.ndarray[np.float64, ndim=1]
        Coefficients of AR model.
    theta: np.ndarray[np.float64, ndim=1]
        Coefficients of MA model.
    c: np.float64
        Constant of the model.
    p: int
        Order of AR(p) model.
    q: int
        Order of MA(q) model.

    Returns
    -------
    u: np.ndarray[np.float64, ndim=1]
        Residual of the model.
    """
    cdef np.float64_t s
    cdef int i, j, t, T = np.size(y)
    cdef np.ndarray[np.float64_t, ndim=1] u = np.zeros([T], dtype=np.float64)

    for t in range(T):
        s = 0
        for i in range(min(t, max(q, p))):
            if i < q:
                s += u[t - i - 1] * theta[i]
            if i < p:
                s += y[t - i - 1] * phi[i]
        u[t] = y[t] - c - s
    return u


cpdef tuple model_ARMA_GARCH_cython(
        np.ndarray[np.float64_t, ndim=1] y,
        np.ndarray[np.float64_t, ndim=1] phi,
        np.ndarray[np.float64_t, ndim=1] theta,
        np.ndarray[np.float64_t, ndim=1] alpha,
        np.ndarray[np.float64_t, ndim=1] beta,
        np.float64_t c, np.float64_t omega,
        int p, int q, int Q, int P
    ):
    """ 
    AutoRegressive Moving Average model of order q and p, such that: 
    y_t = c + phi_1 * y_t-1 + ... + phi_p * y_t-p + theta_1 * u_t-1 + ...
          + theta_q * u_t-q + u_t
    
    With Generalized AutoRegressive Conditional Heteroskedasticity volatility
    model of order Q and P, such that:
    u_t = z_t * h_t 
    h_t^2 = omega + alpha_1 * u^2_t-1 + ... + alpha_Q * u^2_t-Q 
            + beta_1 * h^2_t-1 + ... + beta_P * h^2_t-P
    
    Parameters
    ----------
    y: np.ndarray[np.float64, ndim=1]
        Time series.
    phi: np.ndarray[np.float64, ndim=1]
        Coefficients of AR model.
    theta: np.ndarray[np.float64, ndim=1]
        Coefficients of MA model.
    alpha: np.ndarray[np.float64, ndim=1]
        Coefficients of MA part of GARCH.
    beta: np.ndarray[np.float64, ndim=1]
        Coefficients of AR part of GARCH.
    c: np.float64
        Constant of the model.
    p: int
        Order of AR(p) model.
    q: int
        Order of MA(q) model.
    Q: int
        Order of MA part of GARCH.
    P: int
        Order of AR part of GARCH.

    Returns
    -------
    u: np.ndarray[np.float64, ndim=1]
        Residual of the model. 
    h: np.ndarray[np.float64, ndim=1]
        Conditional volatility of the model. 
    """
    cdef np.float64_t arma, arch
    cdef int i, j, t, T = np.size(y)
    cdef np.ndarray[np.float64_t, ndim=1] u = np.zeros([T], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] h = np.zeros([T], dtype=np.float64)
    
    for t in range(T):
        arma = 0.
        arch = 0.
        for i in range(min(t, max(q, p, Q, P))):
            if i < p:
                arma += y[t-i-1] * phi[i]
            if i < q:
                arma += u[t-i-1] * theta[i]
            if i < Q:
                arch += u[t-i-1]**2 * alpha[i]
            if i < P:
                arch += h[t-i-1]**2 * beta[i]
            if arch < 0.:
                return 1e8 * np.ones([T], dtype=np.float64), np.ones([T], dtype=np.float64)
            if arch > 1e20 or arma > 1e20:
                return 1e6 * np.ones([T], dtype=np.float64), np.ones([T], dtype=np.float64)
        u[t] = y[t] - c - arma
        h[t] = sqrt(omega + arch)
    return u, h


cpdef tuple model_ARMAX_GARCH_cython(
        np.ndarray[np.float64_t, ndim=1] y,
        np.ndarray[np.float64_t, ndim=2] x,
        np.ndarray[np.float64_t, ndim=1] phi,
        np.ndarray[np.float64_t, ndim=1] psi,
        np.ndarray[np.float64_t, ndim=1] theta,
        np.ndarray[np.float64_t, ndim=1] alpha,
        np.ndarray[np.float64_t, ndim=1] beta,
        np.float64_t c, np.float64_t omega,
        int p, int q, int Q, int P
    ):
    """ 
    AutoRegressive Moving Average model of order q and p, such that: 
    y_t = c + phi_1 * y_t-1 + ... + phi_p * y_t-p + psi_t * x_t 
          + theta_1 * u_t-1 + ... + theta_q * u_t-q + u_t
    
    With Generalized AutoRegressive Conditional Heteroskedasticity volatility
    model of order Q and P, such that:
    u_t = z_t * h_t 
    h_t^2 = omega + alpha_1 * u^2_t-1 + ... + alpha_Q * u^2_t-Q 
            + beta_1 * h^2_t-1 + ... + beta_P * h^2_t-P
    
    Parameters
    ----------
    y: np.ndarray[np.float64, ndim=1]
        Time series.
    x: np.ndarray[np.float64, ndim=2]
        Time series of external features.
    phi: np.ndarray[np.float64, ndim=1]
        Coefficients of AR model.
    psi: np.ndarray[np.float64, ndim=1]
        Coefficients of external features.
    theta: np.ndarray[np.float64, ndim=1]
        Coefficients of MA model.
    alpha: np.ndarray[np.float64, ndim=1]
        Coefficients of MA part of GARCH.
    beta: np.ndarray[np.float64, ndim=1]
        Coefficients of AR part of GARCH.
    c: np.float64
        Constant of the model.
    p: int
        Order of AR(p) model.
    q: int
        Order of MA(q) model.
    Q: int
        Order of MA part of GARCH.
    P: int
        Order of AR part of GARCH.

    Returns
    -------
    u: np.ndarray[np.float64, ndim=1]
        Residual of the model. 
    h: np.ndarray[np.float64, ndim=1]
        Conditional volatility of the model. 
    """
    cdef np.float64_t armax, arch
    cdef int i, j, t, T = np.size(y)
    cdef np.ndarray[np.float64_t, ndim=1] u = np.zeros([T], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] h = np.zeros([T], dtype=np.float64)
    
    for t in range(T):
        armax = sum(x[t] * psi)
        arch = 0.
        for i in range(min(t, max(q, p, Q, P))):
            if i < p:
                armax += y[t-i-1] * phi[i]
            if i < q:
                armax += u[t-i-1] * theta[i]
            if i < Q:
                arch += u[t-i-1]**2 * alpha[i]
            if i < P:
                arch += h[t-i-1]**2 * beta[i]
            if arch < 0.:
                return 1e8 * np.ones([T], dtype=np.float64), np.ones([T], dtype=np.float64)
            if arch > 1e20 or armax > 1e20:
                return 1e6 * np.ones([T], dtype=np.float64), np.ones([T], dtype=np.float64)
        u[t] = y[t] - c - armax
        h[t] = sqrt(omega + arch)
    return u, h