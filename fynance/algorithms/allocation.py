#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-09-12 14:52:08
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-12 20:12:47

""" Algorithms of portfolio allocation. """

# Built-in packages

# Third party packages
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

# Local packages

__all__ = ['MVP', 'HRP']

# =========================================================================== #
#                     HRP developed by Marcos Lopez de Prado                  #
# =========================================================================== #


def _corr_dist(mat_corr):
    """ Compute a distance matrix based on correlation.

    Parameters
    ----------
    mat_corr: np.ndarray[ndim=2, dtype=float] or pd.DataFrame
        Matrix correlation.

    Returns
    -------
    mat_dist_corr: np.ndarray[ndim=2, dtype=float] or pd.DataFrame
        Matrix distance correlation.

    """
    return ((1 - mat_corr) / 2.) ** 0.5


def _get_quasi_diag(link):
    """ Compute quasi diagonal matrix.

    TODO : verify the efficiency

    Parameter
    ---------
    link: list of N lists
        Linkage matrix, N list (cluster) of 4-tuple such that the two first
        elements are the costituents, third report the distance between the two
        first, and fourth is the number of element (<= N) in this cluster.

    Returns
    -------
    sortIx: list
        Sorted list of items.

    """
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items

    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i, j = df0.index, df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index

    return sortIx.tolist()


def _get_rec_bisec(mat_cov, sortIx):
    """ Compute weights.

    TODO : verify the efficiency /! must be not efficient /!

    Parameters
    ----------
    mat_cov: pd.DataFrame
        Matrix variance-covariance
    sortIx: list
        Sorted list of items.

    Returns
    -------
    pd.DataFrame
       Weights.

    """
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster

    while len(cItems) > 0:
        cItems = [i[j: k] for i in cItems for j, k in (
            (0, int(len(i) / 2)),
            (int(len(i) / 2), len(i))
        ) if len(i) > 1]  # bi-section

        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = _get_cluster(mat_cov, cItems0)
            cVar1 = _get_cluster(mat_cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2

    return w


def _get_cluster(mat_cov, cItems):
    """ Compute cluster for variance.

    Parameters
    ----------
    mat_cov: pd.DataFrame
        Covariance matrix.
    cItems: list
        Cluster.

    Returns
    -------
    cVar: float
        Cluster variance

    """
    cov_ = mat_cov.loc[cItems, cItems]  # matrix slice
    w_ = _get_IVP(cov_).reshape(-1, 1)
    cVar = ((w_.T @ cov_) @ w_)  # [0, 0]

    return cVar.values[0, 0]


def _get_IVP(mat_cov):
    """ Compute the inverse-variance matrix.

    Parameters
    ----------
    mat_cov: pd.DataFrame
        Matrix covariance.

    Returns
    -------
    ivp: pd.DataFrame
        Matrix of inverse-variance.

    """
    ivp = 1. / np.diag(mat_cov)
    ivp /= ivp.sum()

    return ivp


def HRP(series, method='single', metric='euclidean'):
    """ Compute weights following the Hierarchical Risk Parity algorithm.

    Hierarchical Risk Parity is developed by Marco Lopez de Prado [1]_. First
    step is clustering, second step is allocating weights.

    Parameters
    ----------
    series : pd.DataFrame
        Dataframe of price or return's asset series. Some errors will happen if
        one or more series are constant.
    method, metric: str
        Parameters for linkage algorithm, default ``method='single'`` and
        ``metric='euclidean'``.

    Returns
    -------
    pd.Series
        Vecotr of weights computed by HRP algorithm.

    References
    ----------
    .. [1] https://ssrn.com/abstract=2708678

    """
    # Compute covariance and correlation matrix
    mat_cov = series.cov()
    mat_corr = series.corr()  # .fillna(0)
    # Compute distance matrix
    mat_dist = _corr_dist(mat_corr).fillna(0)
    mat_dist_corr = squareform(mat_dist)
    link = sch.linkage(mat_dist_corr, method=method, metric=metric)
    # Sort linked matrix
    sortIx = _get_quasi_diag(link)
    sortIx = mat_corr.index[sortIx].tolist()
    weights = _get_rec_bisec(mat_cov, sortIx)

    return weights


# =========================================================================== #
#                                    IVP                                      #
# =========================================================================== #


def MVP(series, normalize=True):
    r""" Get weights of the Minimum Variance Portfolio [2]_ allocation.

    The vector of weights noted math:`w` that minimize the portfolio variance
    is define as below:

    .. math::
        w = \frac{\Omega^{-1} e}{e' \Omega^{-1} e}
        with \sum_{i=1}^{N} w_i = 1

    Where math:`\Omega` is the asset's variance-covariance matrix and math:`e`
    is a vector of ones.

    Parameters
    ----------
    series : array_like
        Each column is a time-series of price or return's asset.
    normalize : boolean, optional
        If True normalize the weigths such that math:`0 \leq w_i \leq 1` and
        math:`\sum_{i=1}^{N} w_i = 1`, math:`\forall i`.

    Returns
    -------
    array_like
        Vector of weights to apply to the assets.

    References
    ----------
    .. [2] https://breakingdownfinance.com/finance-topics/modern-portfolio-the\
    ory/minimum-variance-portfolio/

    See Also
    --------
    HRP

    """
    mat_cov = np.cov(series, rowvar=False)
    # Inverse variance matrix
    try:
        iv = np.linalg.inv(mat_cov)

    except np.linalg.LinAlgError:
        iv = np.linalg.pinv(mat_cov)

    e = np.ones([iv.shape[0], 1])
    w = (iv @ e) / (e.T @ iv @ e)

    if normalize:
        w = w - np.min(w)

        return w / np.sum(w)

    return w
