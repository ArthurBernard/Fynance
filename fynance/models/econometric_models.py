#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
import pandas as pd

# Internal packages
from .econometric_models_cy import *

__all__ = [
    'get_parameters', 'MA', 'ARMA', 'ARMA_GARCH', 'ARMAX_GARCH'
]

#=============================================================================#
#                             PARAMETERS FUNCTION                             #
#=============================================================================#


def get_parameters(params, p=0, q=0, Q=0, P=0, cons=True):
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
    i = 0
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


def MA(y, theta, c, q):
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
    # Set type of variables
    if isinstance(y, pd.DataFrame) or isinstance(y, list) or isinstance(y, pd.Series):
        y = np.asarray(y)
    y = y.astype(np.float64).reshape([y.size])
    if isinstance(theta, list):
        theta = np.asarray(theta)
    theta = theta.astype(np.float64).reshape([theta.size])
    # Compute residuals
    u = MA_cy(y, theta, np.float64(c), int(q))
    return u


def ARMA(y, phi, theta, c, p, q):
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
    # Set type variables and parameters
    y = np.asarray(y, dtype=np.float64).reshape([y.size])
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    # Compute residuals
    u = ARMA_cy(y, phi, theta, float(c), int(p), int(q))
    return u


def ARMA_GARCH(y, phi, theta, alpha, beta, c, omega, p, q, Q, P):
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
    y = np.asarray(y, dtype=np.float64).reshape([y.size])
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    u, h = ARMA_GARCH_cy(
        y, phi, theta, alpha, beta, float(c), float(omega), int(p), int(q), 
        int(Q), int(P)
    )
    return u, h


def ARMAX_GARCH(y, x, phi, psi, theta, alpha, beta, c, omega, p, q, Q, P):
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
    # Set array variables
    y = np.asarray(y, dtype=np.float64).reshape([y.size])
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    # Compute residuals and volatility
    u, h = ARMAX_GARCH_cy(
        y, x, phi, theta, psi, alpha, beta, float(c), float(omega), int(p), 
        int(q), int(Q), int(P)
    )
    return u, h