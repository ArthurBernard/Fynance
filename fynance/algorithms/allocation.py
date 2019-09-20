#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-09-12 14:52:08
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-20 17:29:59

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

__all__ = ['ERC', 'HRP', 'IVP', 'MDP', 'MVP', 'MVP_uc', 'rolling_allocation']


# =========================================================================== #
#                         Equal Risk Contribution                             #
# =========================================================================== #


def ERC(X, w0=None, up_bound=1., low_bound=0.):
    r""" Get weights of Equal Risk Contribution portfolio allocation.

    Notes
    -----
    Weights of Equal Risk Contribution, as described by S. Maillard, T.
    Roncalli and J. Teiletche [1]_, verify the following problem:

    .. math::
        w = \text{arg min } f(w) \\
        u.c. \begin{cases}w'e = 1 \\
                          0 \leq w_i \leq 1 \\
             \end{cases}

    With:

    .. math::
        f(w) = N \sum_{i=1}^{N}w_i^2 (\Omega w)_i^2
        - \sum_{i,j=1}^{N} w_i w_j (\Omega w)_i (\Omega w)_j

    Where :math:`\Omega` is the variance-covariance matrix of `X` and :math:`N`
    the number of assets.

    Parameters
    ----------
    X : array_like
        Each column is a series of price or return's asset.
    w0 : array_like, optional
        Initial weights to maximize.
    up_bound, low_bound : float, optional
        Respectively maximum and minimum values of weights, such that low_bound
        :math:`\leq w_i \leq` up_bound :math`\forall i`. Default is 0 and 1.

    Returns
    -------
    array_like
        Weights that minimize the Equal Risk Contribution portfolio.

    References
    ----------
    .. [1] http://thierry-roncalli.com/download/erc-slides.pdf

    """
    T, N = X.shape
    SIGMA = np.cov(X, rowvar=False)
    up_bound = max(up_bound, 1 / N)

    def f_ERC(w):
        w = w.reshape([N, 1])
        arg = N * np.sum(w ** 2 * (SIGMA @ w) ** 2)

        return arg - np.sum(w * (SIGMA @ w) * np.sum(w * (SIGMA @ w)))

    # Set inital weights
    if w0 is None:
        w0 = np.ones([N]) / N

    const_sum = LinearConstraint(np.ones([1, N]), [1], [1])
    const_ind = Bounds(low_bound * np.ones([N]), up_bound * np.ones([N]))
    result = minimize(
        f_ERC,
        w0,
        method='SLSQP',
        constraints=[const_sum],
        bounds=const_ind
    )

    return result.x.reshape([N, 1])


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


def HRP(X, method='single', metric='euclidean', low_bound=0., up_bound=1.0):
    r""" Get weights of the Hierarchical Risk Parity allocation.

    Notes
    -----
    Hierarchical Risk Parity algorithm is developed by Marco Lopez de Prado
    [2]_. First step is clustering and second step is allocating weights.

    Parameters
    ----------
    X : array_like
        Each column is a price or return's asset series. Some errors will
        happen if one or more series are constant.
    method, metric: str
        Parameters for linkage algorithm, default ``method='single'`` and
        ``metric='euclidean'``.
    low_bound, up_bound : float
        Respectively minimum and maximum value of weights, such that low_bound
        :math:`\leq w_i \leq` up_bound :math`\forall i`. Default is 0 and 1.

    Returns
    -------
    np.ndarray
        Vecotr of weights computed by HRP algorithm.

    References
    ----------
    .. [2] https://ssrn.com/abstract=2708678

    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    idx = X.columns
    up_bound = max(up_bound, 1 / X.shape[1])
    low_bound = min(low_bound, 1 / X.shape[1])

    # Compute covariance and correlation matrix
    mat_cov = X.cov()
    mat_corr = X.corr().fillna(0)
    # Compute distance matrix
    # print(mat_corr)
    mat_dist = _corr_dist(mat_corr).fillna(0)
    mat_dist_corr = squareform(mat_dist)
    link = sch.linkage(mat_dist_corr, method=method, metric=metric)
    # Sort linked matrix
    sortIx = _get_quasi_diag(link)
    sortIx = mat_corr.index[sortIx].tolist()
    w = _get_rec_bisec(mat_cov, sortIx)
    w = w.loc[idx].to_numpy(copy=True).reshape([w.size, 1])

    return _normalize(w, up_bound=up_bound, low_bound=low_bound)


# =========================================================================== #
#                        Inverse Variance Portfolio                           #
# =========================================================================== #


def IVP(X, normalize=False, low_bound=0., up_bound=1.0):
    r""" Get weights of the Inverse Variance Portfolio allocation.

    Notes
    -----
    w are computed by the inverse of the asset's variance [3]_ such that:

    .. math::
        w_i = \frac{1}{\sigma_k^2} (\sum_{i} \frac{1}{\sigma_i^2})^{-1}

    With :math:`\sigma_i^2` is the variance of asset i.

    Parameters
    ----------
    X : array_like
        Each column is a price or return's asset series.
    normalize : bool, optional
        If True normalize the weights such that :math:`\sum_{i=1}^{N} w_i = 1`
        and :math:`0 \leq w_i \leq 1`. Default is False.
    low_bound, up_bound : float, optional
        Respectively minimum and maximum values of weights, such that low_bound
        :math:`\leq w_i \leq` up_bound :math`\forall i`. Default is 0 and 1.

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
    up_bound = max(up_bound, 1 / X.shape[1])
    low_bound = min(low_bound, 1 / X.shape[1])

    if normalize:
        w = w - np.min(w)
        w = w / np.sum(w)

    #    return w.reshape([mat_cov.shape[0], 1])
    w = _normalize(w, up_bound=up_bound, low_bound=low_bound)
    # w = w * (up_bound - low_bound) + low_bound

    return w.reshape([mat_cov.shape[0], 1])


# =========================================================================== #
#                         Maximum Variance Portfolio                          #
# =========================================================================== #


def MVP(X, normalize=False):
    r""" Get weights of the Minimum Variance Portfolio allocation.

    Notes
    -----
    The vector of weights noted :math:`w` that minimize the portfolio variance
    [4]_ is define as below:

    .. math:: w = \frac{\Omega^{-1} e}{e' \Omega^{-1} e} \\
    .. math:: \text{With } \sum_{i=1}^{N} w_i = 1

    Where :math:`\Omega` is the asset's variance-covariance matrix and
    :math:`e` is a vector of ones.

    Parameters
    ----------
    X : array_like
        Each column is a time-series of price or return's asset.
    normalize : boolean, optional
        If True normalize the weigths such that :math:`0 \leq w_i \leq 1` and
        :math:`\sum_{i=1}^{N} w_i = 1`, :math:`\forall i`. Default is False.

    Returns
    -------
    array_like
        Vector of weights to apply to the assets.

    References
    ----------
    .. [4] https://breakingdownfinance.com/finance-topics/modern-portfolio-theory/minimum-variance-portfolio/

    See Also
    --------
    HRP

    """
    mat_cov = np.cov(X, rowvar=False)
    # Inverse variance matrix
    try:
        iv = np.linalg.inv(mat_cov)

    except np.linalg.LinAlgError:
        try:
            iv = np.linalg.pinv(mat_cov)
        except np.linalg.LinAlgError:
            display(mat_cov)
            raise np.linalg.LinAlgError

    e = np.ones([iv.shape[0], 1])
    w = (iv @ e) / (e.T @ iv @ e)

    if normalize:
        w = w - np.min(w)

        return w / np.sum(w)

    return w


def MVP_uc(X, w0=None, up_bound=1., low_bound=0.):
    r""" Get weights of the Minimum Variance Portfolio under constraints.

    Notes
    -----
    Weights of Minimum Variance Portfolio verify the following problem:

    .. math::
        w = \text{arg min } w' \Omega w \\
        u.c. \begin{cases}w'e = 1 \\
                          0 \leq w_i \leq 1 \\
             \end{cases}

    Where :math:`\Omega` is the variance-covariance matrix of `X` and :math:`e`
    a vector of ones.

    Parameters
    ----------
    X : array_like
        Each column is a series of price or return's asset.
    w0 : array_like, optional
        Initial weights to maximize.
    up_bound, low_bound : float, optional
        Respectively maximum and minimum values of weights, such that low_bound
        :math:`\leq w_i \leq` up_bound :math`\forall i`. Default is 0 and 1.

    Returns
    -------
    array_like
        Weights that minimize the variance of the portfolio.

    """
    mat_cov = np.cov(X, rowvar=False)
    N = X.shape[1]
    up_bound = max(up_bound, 1 / N)

    def f_MVP(w):
        w = w.reshape([N, 1])
        return w.T @ mat_cov @ w

    # Set inital weights
    if w0 is None:
        w0 = np.ones([N]) / N

    # Set constraints and minimze
    const_sum = LinearConstraint(np.ones([1, N]), [1], [1])
    const_ind = Bounds(low_bound * np.ones([N]), up_bound * np.ones([N]))
    result = minimize(
        f_MVP,
        w0,
        method='SLSQP',
        constraints=[const_sum],
        bounds=const_ind
    )

    return result.x.reshape([N, 1])


# =========================================================================== #
#    Maximum Diversification Portfolio developed by Choueifaty and Coignard   #
# =========================================================================== #


def MDP(X, w0=None, up_bound=1., low_bound=0.):
    r""" Get weights of Maximum Diversified Portfolio allocation.

    Notes
    -----
    Weights of Maximum Diversification Portfolio, as described by Y. Choueifaty
    and Y. Coignard [5]_, verify the following problem:

    .. math::

        w = \text{arg max } D(w) \\
        u.c. \begin{cases}w'e = 1 \\
                          0 \leq w_i \leq 1 \\
             \end{cases}

    Where :math:`D(w)` is the diversified ratio of portfolio weighted by `w`.

    Parameters
    ----------
    X : array_like
        Each column is a series of price or return's asset.
    w0 : array_like, optional
        Initial weights to maximize.
    up_bound, low_bound : float, optional
        Respectively maximum and minimum values of weights, such that low_bound
        :math:`\leq w_i \leq` up_bound :math`\forall i`. Default is 0 and 1.

    Returns
    -------
    array_like
        Weights that maximize the diversified ratio of the portfolio.

    See Also
    --------
    diversified_ratio

    References
    ----------
    .. [5] tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf

    """
    T, N = X.shape
    up_bound = max(up_bound, 1 / N)

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


def rolling_allocation(f, X, n=252, s=63, ret=True, drift=True, **kwargs):
    r""" Roll an algorithm of portfolio allocation.

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
    X = pd.DataFrame(X).fillna(method='ffill')
    idx = X.index
    w_mat = pd.DataFrame(index=idx, columns=X.columns)
    portfolio = pd.Series(100., index=idx, name='portfolio')

    if ret:
        X_ = X.pct_change()

    else:
        X_ = X

    roll = _RollingMechanism(idx, n=n, s=s)

    def process(series):
        # True if less than 50% of obs. are constant
        return series.value_counts(dropna=False).max() < 0.5 * n

    for slice_n, slice_s in roll():
        # Select X
        sub_X = X_.loc[slice_n].copy()
        assets = list(X.columns[sub_X.apply(process)])
        sub_X = sub_X.fillna(method='bfill')
        # Compute weights
        if len(assets) == 1:
            w = np.array([[1.]])

        else:
            w = f(sub_X.loc[:, assets].values, **kwargs)

        w_mat.loc[roll.d, assets] = w.flatten()
        w_mat.loc[roll.d, :] = w_mat.loc[roll.d, :].fillna(0.)
        # Compute portfolio performance
        perf = _perf_alloc(
            X.loc[slice_s, assets].fillna(method='bfill').values,
            w=w,
            drift=drift
        )
        portfolio.loc[slice_s] = portfolio.loc[roll.d] * perf.flatten()

    w_mat = w_mat.fillna(method='ffill').fillna(0.)

    return portfolio, w_mat


# =========================================================================== #
#                                   Tools                                     #
# =========================================================================== #


def _perf_alloc(X, w, drift=True):
    # Compute portfolio performance following specified weights
    if w.ndim == 1 and not isinstance(w, pd.Series):
        w = w.reshape([w.size, 1])

    if drift:
        return (X / X[0, :]) @ w

    perf = np.zeros(X.shape)
    perf[1:] = (X[1:] / X[:-1] - 1)

    return np.cumprod(perf @ w + 1)


def _normalize(w, low_bound=0., up_bound=1., sum_w=1., max_iter=1000):
    # Iterative algorithm to set bounds
    if up_bound < sum_w / w.size or low_bound > sum_w / w.size:

        raise ValueError('Low or up bound exceeded sum weight constraint.')

    j = 0
    while (min(w) < low_bound or max(w) > up_bound) and j < max_iter:
        for i in range(w.size):
            w[i] = min(w[i], up_bound)
            w[i] = max(w[i], low_bound)

        w = sum_w * (w / sum(w))
        j += 1

    if j >= max_iter:
        print('Iterative normalize algorithm exceeded max iterations')

    return w
