#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-09-12 14:52:08
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-16 10:44:47

""" Algorithms of portfolio allocation. """

# Built-in packages

# Third party packages
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.optimize import Bounds, LinearConstraint, minimize

# Local packages
from fynance.tools.metrics import diversified_ratio
from .rolling import _RollingMechanism

# TODO : cython

__all__ = ['HRP', 'IVP', 'MDP', 'MVP', 'rolling_allocation']

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
    mat_cov : array_like
        Variance-covariance matrix.

    Returns
    -------
    pd.DataFrame
        Matrix of inverse-variance.

    """
    ivp = 1. / np.diag(mat_cov)
    ivp /= np.sum(ivp)

    return ivp


def HRP(X, method='single', metric='euclidean'):
    """ Get weights of the Hierarchical Risk Parity allocation.

    Notes
    -----
    Hierarchical Risk Parity algorithm is developed by Marco Lopez de Prado
    [1]_. First step is clustering and second step is allocating weights.

    Parameters
    ----------
    X : array_like
        Each column is a price or return's asset series. Some errors will
        happen if one or more series are constant.
    method, metric: str
        Parameters for linkage algorithm, default ``method='single'`` and
        ``metric='euclidean'``.

    Returns
    -------
    np.ndarray
        Vecotr of weights computed by HRP algorithm.

    References
    ----------
    .. [1] https://ssrn.com/abstract=2708678

    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Compute covariance and correlation matrix
    mat_cov = X.cov()
    mat_corr = X.corr()  # .fillna(0)
    # Compute distance matrix
    mat_dist = _corr_dist(mat_corr).fillna(0)
    mat_dist_corr = squareform(mat_dist)
    link = sch.linkage(mat_dist_corr, method=method, metric=metric)
    # Sort linked matrix
    sortIx = _get_quasi_diag(link)
    sortIx = mat_corr.index[sortIx].tolist()
    weights = _get_rec_bisec(mat_cov, sortIx)

    return weights.to_numpy(copy=True).reshape([weights.size, 1])


# =========================================================================== #
#                        Inverse Variance Portfolio                           #
# =========================================================================== #


def IVP(X, normalize=False):
    r""" Get weights of the Inverse Variance Portfolio allocation.

    Notes
    -----
    Weights are computed by the inverse of the asset's variance [3]_ such that:

    .. math::
        w_i = \frac{1}{\sigma_k^2} (\sum_{i} \frac{1}{\sigma_i^2})^{-1}

    With :math:`\sigma_i^2` is the variance of asset i.

    Parameters
    ----------
    X : array_like
        Each column is a price or return's asset series.
    normalize : bool, optional
        If True normalize the weights such that :math:`\sum_{i=1}^{N} w_i = 1`
        and :math:`0 \leq w_i \leq 1`.

    Returns
    -------
    np.ndarray
        Vector of weights computed by the IVP algorithm.

    References
    ----------
    .. [3] https://en.wikipedia.org/wiki/Inverse-variance_weighting

    """
    mat_cov = np.cov(X, rowvar=False)
    w = _get_IVP(mat_cov)

    if normalize:
        w = w - np.min(w)

        return w / np.sum(w)

    return w.reshape([mat_cov.shape[0], 1])


# =========================================================================== #
#                         Maximum Variance Portfolio                          #
# =========================================================================== #


def MVP(X, normalize=True):
    r""" Get weights of the Minimum Variance Portfolio allocation.

    Notes
    -----
    The vector of weights noted :math:`w` that minimize the portfolio variance
    [2]_ is define as below:

    .. math::
        w = \frac{\Omega^{-1} e}{e' \Omega^{-1} e}
        \text{ with } \sum_{i=1}^{N} w_i = 1

    Where :math:`\Omega` is the asset's variance-covariance matrix and :math:`e`
    is a vector of ones.

    Parameters
    ----------
    X : array_like
        Each column is a time-series of price or return's asset.
    normalize : boolean, optional
        If True normalize the weigths such that :math:`0 \leq w_i \leq 1` and
        :math:`\sum_{i=1}^{N} w_i = 1`, :math:`\forall i`.

    Returns
    -------
    array_like
        Vector of weights to apply to the assets.

    References
    ----------
    .. [2] https://breakingdownfinance.com/finance-topics/modern-portfolio-theory/minimum-variance-portfolio/

    See Also
    --------
    HRP

    """
    mat_cov = np.cov(X, rowvar=False)
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


# =========================================================================== #
#    Maximum Diversification Portfolio developed by Choueifaty and Coignard   #
# =========================================================================== #


def MDP(X, w0=None, up_bound=1., low_bound=0.):
    """ Get weights of Maximum Diversified Portfolio allocation.

    Parameters
    ----------
    X : array_like
        Each column is a series of price or return's asset.
    w0 : array_like, optional
        Initial weights to maximize.
    up_bound, low_bound : float, optional
        Respectively maximum and minimum values of weights.

    Returns
    -------
    array_like
        Weights that maximize the diversified ratio of the portfolio.

    References
    ----------
    .. [2] tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf

    """
    T, N = X.shape

    # Set function to minimze
    def f_max_divers_weights(w):
        return - diversified_ratio(X, w=w).flatten()

    # Set inital weights
    if w0 is None:
        w0 = np.ones([N]) / N

    # Set constraints and minimze
    const_sum = LinearConstraint(np.ones([1, N]), [1], [1])
    const_ind = Bounds(low_bound * np.ones([N]), up_bound * np.ones([N]))
    result = minimize(
        f_max_divers_weights,
        w0,
        method='SLSQP',
        constraints=[const_sum],
        bounds=const_ind
    )

    return result.x.reshape([N, 1])


# =========================================================================== #
#                             Rolling allocation                              #
# =========================================================================== #


def _perf_alloc(X, w, drift=True):
    if w.ndim == 1 and not isinstance(w, pd.Series):
        w = w.reshape([w.size, 1])

    if drift:
        return (X / X[0, :]) @ w

    perf = np.zeros(X.shape)
    perf[1:] = (X[1:] / X[:-1] - 1)

    return np.cumprod(perf @ w + 1)


def rolling_allocation(f, X, n=252, s=63, ret=True, drift=True, **kwargs):
    r""" Roll an algorithm of allocation.

    Notes
    -----
    Weights are computed on the past data from ``t - n`` to ``t`` and are
    applied to backtest on data from ``t`` to ``t + s``.

    .. math::
        \forall t \in [n, T], w_{t:t+s} = f(X_{t-n:t})

    Parameters
    ----------
    f : callable
        Allocation algorithm that take as parameters a subarray of ``X``
        and ``**kwargs``, and return a vector (as ``np.ndarray``) of weights.
    X : array_like
        Data matrix, each columns is a series of prices, indexes or
        performances, each row is a observation at time ``t``.
    n, s : int
        Respectively the number of observations to compute weights and the
        number of observations to roll. Default is ``n=252`` and ``s=63``.
    ret : bool, optional
        If True (default) pass to ``f`` the returns of ``X``. Otherwise pass
        ``X`` to ``f``.
    drift : bool, optional
        If False performance of the portfolio is computed as if we rebalance
        the weights of asset at each timeframe. Otherwise we let to drift the
        weights. Default is True.
    **kwargs
        Any keyword arguments to pass to ``f``.

    Returns
    -------
    pd.Series
        Performance of the portfolio allocated following ``f`` algorithm.
    pd.DataFrame
        Weights of the portfolio allocated following ``f`` algorithm.

    """
    X = pd.DataFrame(X)
    idx = X.index
    w_mat = pd.DataFrame(index=idx, columns=X.columns)
    portfolio = pd.Series(100., index=idx, name='portfolio')

    if ret:
        X_ = X.pct_change()

    else:
        X_ = X

    roll = _RollingMechanism(idx, n=n, s=s)

    for slice_n, slice_s in roll():
        # Select X
        sub_X = X_.loc[slice_n].copy()
        sub_X = sub_X.dropna(axis=1, how='all').fillna(method='bfill')
        assets = sub_X.columns
        # Compute weights
        w = f(sub_X.values, **kwargs)
        w_mat.loc[roll.d, assets] = w.flatten()
        # Compute portfolio performance
        perf = _perf_alloc(X.loc[slice_s, assets].values, w=w, drift=drift)
        portfolio.loc[slice_s] = portfolio.loc[roll.d] * perf.flatten()

    w_mat = w_mat.fillna(method='ffill').fillna(0.)

    return portfolio, w_mat
