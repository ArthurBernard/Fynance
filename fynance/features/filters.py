#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-28 18:12:52
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-18 15:26:31

# Built-in packages

# External packages
import numpy as np

# Local packages


__all__ = ['kalman']


def kalman(X, distribution='normal'):
    """ Compute the Kalman filter.

    Kalman filter is computed as described in the paper by G. Welch and
    G. Bishop [1]_.

    Parameters
    ----------
    X : array_like
        Observed data.
    distribution : str, optional
        An available distribution in scipy library.

    Returns
    -------
    array_like
        Filter of kalman following the given distribution.

    References
    ----------
    .. [1] https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

    """
    T, n = X.shape

    # Initial estimation
    # TODO : Check better initialization
    m_0 = 0
    C_0 = 1

    # Set variables
    # TODO : Check shape of variables
    a = np.zeros([T, n])
    R = np.zeros([T, n])

    for t in range(1, T):
        # Time update (predict)
        a[t] = G[t] @ m[t - 1]
        R[t] = G[t] @ C[t - 1] @ G[t].T + W[t]

        # Measurement update (correct)
        A[t] = R[t] @ F[t] @ np.linalg.pinv(F[t] @ R[t] @ F[t].T + V[t])
        m[t] = a[t] + A[t] @ (X[t] - F[t].T @ a[t])
        C[t] = (np.identity(n) - A[t] @ F[t]) @ R[t]

    s_0 = 0
    x_0 = X[0]

    x_hat_1
