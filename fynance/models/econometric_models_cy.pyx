#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin
cimport numpy as np
from libc.math cimport sqrt, pi, log
from cpython cimport bool

__all__ = [
    'get_parameters_cy', 'MA_cy', 'ARMA_cy', 'ARMA_GARCH_cy', 'ARMAX_GARCH_cy',
]

#=============================================================================#
#                             PARAMETERS FUNCTION                             #
#=============================================================================#


cpdef tuple get_parameters_cy(
        np.ndarray[np.float64_t, ndim=1] params,
        int p=0, int q=0, int Q=0, int P=0, 
        bool cons=True
    ):
    """ Get parameters for ARMA-GARCH models 
    
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

    See also
    --------
    ARMAX_GARCH, ARMA_GARCH, ARMA and MA.
    
    """
    cdef int i = 0
    cdef np.ndarray[np.float64_t, ndim=1] phi, theta, alpha, beta
    cdef np.float64_t c, omega

    if cons:
        c = params[i]
        i += 1
    else:
        c = 0.
    if p > 0:
        phi = params[i: p+i]
        i += p
    else:
        phi = np.array([0.], dtype=np.float64)
    if q > 0:
        theta = params[i: q+i]
        i += q
    else: 
        theta = np.array([0.], dtype=np.float64)
    if Q > 0 or P > 0:
        omega = params[i]
        i += 1
        if Q > 0:
            alpha = params[i: Q+i]
            i += Q
        else:
            alpha = np.array([0.], dtype=np.float64)
        if P > 0:
            beta = params[i: P+i]
            i += P
        else:
            beta = np.array([0.], dtype=np.float64)
    else:
        omega = 0.
        alpha = np.array([0.], dtype=np.float64)
        beta = np.array([0.], dtype=np.float64)
    return phi, theta, alpha, beta, c, omega


#=============================================================================#
#                                   MODELs                                    #
#=============================================================================#


cpdef np.ndarray[np.float64_t, ndim=1] MA_cy(
        np.ndarray[np.float64_t, ndim=1] y, 
        np.ndarray[np.float64_t, ndim=1] theta,
        np.float64_t c, int q
    ):
    """ Moving Average model of order `q` s.t: 
    
    .. math:: y_t = c + \\theta_1 * u_{t-1} + ... + \\theta_q * u_{t-q} + u_t 
    
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

    See also
    --------
    ARMA_GARCH, ARMA and ARMAX_GARCH

    """
    cdef np.float64_t s
    cdef int i, t, T = y.size
    cdef np.ndarray[np.float64_t, ndim=1] u = np.zeros([T], dtype=np.float64)  # Residuals
    
    for t in range(T):
        s = 0.
        for i in range(min(t, q)):
            s += u[t-i-1] * theta[i]
        if s > 1e12 or s < -1e12:
            return 1e6 * np.ones([T], dtype=np.float64)
        u[t] = y[t] - c - s
    return u


cpdef np.ndarray[np.float64_t, ndim=1] ARMA_cy(
        np.ndarray[np.float64_t, ndim=1] y,
        np.ndarray[np.float64_t, ndim=1] phi,
        np.ndarray[np.float64_t, ndim=1] theta,
        np.float64_t c, int p, int q
    ):
    """ AutoRegressive Moving Average model of order `q` and `p` s.t: 
    
    .. math:: 

        y_t = c + \\phi_1 * y_{t-1} + ... + \\phi_p * y_{t-p} 
        + \\theta_1 * u_{t-1} + ... + \\theta_q * u_{t-q} + u_t
    
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

    See also
    --------
    ARMA_GARCH, ARMAX_GARCH and MA.
    
    """
    cdef np.float64_t s
    cdef int i, j, t, T = np.size(y)
    cdef np.ndarray[np.float64_t, ndim=1] u = np.zeros([T], dtype=np.float64)

    for t in range(T):
        s = 0.
        for i in range(min(t, max(q, p))):
            if i < q:
                s += u[t - i - 1] * theta[i]
            if i < p:
                s += y[t - i - 1] * phi[i]
            if s > 1e12 or s < -1e12:
                return 1e6 * np.ones([T], dtype=np.float64)
        u[t] = y[t] - c - s
    return u


cpdef tuple ARMA_GARCH_cy(
        np.ndarray[np.float64_t, ndim=1] y,
        np.ndarray[np.float64_t, ndim=1] phi,
        np.ndarray[np.float64_t, ndim=1] theta,
        np.ndarray[np.float64_t, ndim=1] alpha,
        np.ndarray[np.float64_t, ndim=1] beta,
        np.float64_t c, np.float64_t omega,
        int p, int q, int Q, int P
    ):
    """ AutoRegressive Moving Average model of order q and p, such that: 
    
    .. math:: 

        y_t = c + \\phi_1 * y_{t-1} + ... + \\phi_p * y_{t-p} 
        + \\theta_1 * u_{t-1} + ... + \\theta_q * u_{t-q} + u_t
    
    With Generalized AutoRegressive Conditional Heteroskedasticity volatility
    model of order `Q` and `P`, such that:
    
    .. math:: 
        u_t = z_t * h_t

        h_t^2 = \\omega + \\alpha_1 * u^2_{t-1} + ... + \\alpha_Q * u^2_{t-Q}
        + \\beta_1 * h^2_{t-1} + ... + \\beta_P * h^2_{t-P}
    
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

    See also
    --------
    ARMAX_GARCH, ARMA and MA.
    
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
            if arch > 1e12 or arma > 1e12 or arma < -1e12:
                return 1e6 * np.ones([T], dtype=np.float64), np.ones([T], dtype=np.float64)
        u[t] = y[t] - c - arma
        h[t] = sqrt(omega + arch)
    return u, h


cpdef tuple ARMAX_GARCH_cy(
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
    """ AutoRegressive Moving Average model of order q and p, such that: 
    
    .. math:: 

        y_t = c + \\phi_1 * y_{t-1} + ... + \\phi_p * y_{t-p} + \\psi_t * x_t 
        + \\theta_1 * u_{t-1} + ... + \\theta_q * u_{t-q} + u_t
    
    With Generalized AutoRegressive Conditional Heteroskedasticity volatility
    model of order `Q` and `P`, such that:
    
    .. math:: 
        u_t = z_t * h_t

        h_t^2 = \\omega + \\alpha_1 * u^2_{t-1} + ... + \\alpha_Q * u^2_{t-Q}
        + \\beta_1 * h^2_{t-1} + ... + \\beta_P * h^2_{t-P}
    
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
        Constant of the model.
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

    See also
    --------
    ARMA_GARCH, ARMA and MA.
    
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
            if arch > 1e12 or armax > 1e12 or armax < -1e12:
                return 1e6 * np.ones([T], dtype=np.float64), np.ones([T], dtype=np.float64)
        u[t] = y[t] - c - armax
        h[t] = sqrt(omega + arch)
    return u, h