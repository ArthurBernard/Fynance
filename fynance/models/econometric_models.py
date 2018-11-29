#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import fmin

from .econometric_models_cy import *

__all__ = [
    'get_parameters', 'MA', 'ARMA', 'ARMA_GARCH', 'ARMAX_GARCH'
]

#=============================================================================#
#                             PARAMETERS FUNCTION                             #
#=============================================================================#


def get_parameters(params, p=0, q=0, Q=0, P=0, cons=True):
    """
    Get parameters for ARMA-GARCH models 
    
    Parameters
    ----------
    :params: np.ndarray[np.float64, ndim=1]
        Array of model parameters.
    :p, q, Q, P: int
        Order of model.
    :cons: bool
        True if model contains constant.

    Returns
    -------
    :phi: np.ndarray[np.float64, ndim=1]
        AR parameters.
    :theta: np.ndarray[np.float64, ndim=1]
        MA parameters.
    :alpha: np.ndarray[np.float64, ndim=1]
        First part GARCH parameters.
    :beta: np.ndarray[np.float64, ndim=1]
        Last part GARCH parameters.
    :c and omega: float
        Constants of model.
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
    if isinstance(y, pd.DataFrame) or isinstance(y, list) or isinstance(y, pd.Series):
        y = np.asarray(y, dtype=np.float64).reshape([y.size])
    if isinstance(theta, list):
        theta = np.asarray(theta, dtype=np.float64)
    u = MA_cy(y, theta, float(c), int(q))
    return u


def ARMA(y, phi, theta, c, p, q):
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
    if isinstance(y, pd.DataFrame) or isinstance(y, list) or isinstance(y, pd.Series):
        y = np.asarray(y, dtype=np.float64).reshape([y.size])
    if isinstance(theta, list):
        theta = np.asarray(theta, dtype=np.float64)
    if isinstance(phi, list):
        phi = np.asarray(phi, dtype=np.float64)
    u = ARMA_cy(y, phi, theta, float(c), int(p), int(q))
    return u


def ARMA_GARCH(y, phi, theta, alpha, beta, c, omega, p, q, Q, P):
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
        Constant of ARMA model.
    omega: np.float64
        Constant of GARCH model.
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
    if isinstance(y, pd.DataFrame) or isinstance(y, list) or isinstance(y, pd.Series):
        y = np.asarray(y, dtype=np.float64).reshape([y.size])
    if isinstance(theta, list):
        theta = np.asarray(theta, dtype=np.float64)
    if isinstance(phi, list):
        phi = np.asarray(phi, dtype=np.float64)
    if isinstance(alpha, list):
        alpha = np.asarray(alpha, dtype=np.float64)
    if isinstance(beta, list):
        beta = np.asarray(beta, dtype=np.float64)
    u, h = ARMA_GARCH_cy(
        y, phi, theta, alpha, beta, float(c), float(omega), int(p), int(q), 
        int(Q), int(P)
    )
    return u, h


def ARMAX_GARCH(y, x, phi, psi, theta, alpha, beta, c, omega, p, q, Q, P):
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
    if isinstance(y, pd.DataFrame) or isinstance(y, list) or isinstance(y, pd.Series):
        y = np.asarray(y, dtype=np.float64).reshape([y.size])
    if isinstance(x, pd.DataFrame) or isinstance(x, list) or isinstance(x, pd.Series):
        x = np.asarray(x, dtype=np.float64)
    if isinstance(theta, list):
        theta = np.asarray(theta, dtype=np.float64)
    if isinstance(phi, list):
        phi = np.asarray(phi, dtype=np.float64)
    if isinstance(psi, list):
        psi = np.asarray(psi, dtype=np.float64)
    if isinstance(alpha, list):
        alpha = np.asarray(alpha, dtype=np.float64)
    if isinstance(beta, list):
        beta = np.asarray(beta, dtype=np.float64)
    u, h = ARMAX_GARCH_cy(
        y, x, phi, theta, psi, alpha, beta, float(c), float(omega), int(p), 
        int(q), int(Q), int(P)
    )
    return u, h