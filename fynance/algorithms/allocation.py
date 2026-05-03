#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-09-12 14:52:08
# @Last modified by: ArthurBernard
# @Last modified time: 2019-11-05 17:20:52

""" Algorithms of portfolio allocation. """

from __future__ import annotations

# Built-in packages
from typing import Callable

# Third-party
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize

# Local packages
from fynance.features.metrics import diversified_ratio

from .rolling import _RollingMechanism

# TODO : cython

__all__ = ['ERC', 'HRP', 'IVP', 'MDP', 'MVP', 'MVP_uc', 'rolling_allocation']


# =========================================================================== #
#                         Equal Risk Contribution                             #
# =========================================================================== #


def ERC(
    X: NDArray[np.float64],
    w0: NDArray[np.float64] | None = None,
    up_bound: float = 1.,
    low_bound: float = 0.,
) -> NDArray[np.float64]:
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
        :math:`\leq w_i \leq` up_bound :math:`\forall i`. Default is 0 and 1.

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


def _get_quasi_diag(link: NDArray[np.float64]) -> list[int]:
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
    numItems = link[-1, 3]  # number of original leaf items
    # Use a plain list to avoid pandas dtype coercion issues
    items = [int(link[-1, 0]), int(link[-1, 1])]

    while max(items) >= numItems:
        expanded = []
        for item in items:
            if item >= numItems:
                cluster_idx = item - numItems
                expanded.append(int(link[cluster_idx, 0]))
                expanded.append(int(link[cluster_idx, 1]))
            else:
                expanded.append(item)
        items = expanded

    return items


def _get_rec_bisec(mat_cov: NDArray[np.float64], sortIx: list[int]) -> NDArray[np.float64]:
    """ Compute weights via recursive bisection.

    Parameters
    ----------
    mat_cov: np.ndarray
        Matrix variance-covariance (N x N).
    sortIx: list or np.ndarray of int
        Sorted list of asset indices (0..N-1).

    Returns
    -------
    np.ndarray
        Weight vector of shape (N,) indexed by sortIx order.

    """
    n = len(sortIx)
    w = np.ones(n)
    cItems = [list(range(n))]

    while len(cItems) > 0:
        cItems = [i[j: k] for i in cItems for j, k in (
            (0, int(len(i) / 2)),
            (int(len(i) / 2), len(i))
        ) if len(i) > 1]

        for i in range(0, len(cItems), 2):
            cItems0_idx = cItems[i]
            cItems1_idx = cItems[i + 1]
            cItems0 = [sortIx[j] for j in cItems0_idx]
            cItems1 = [sortIx[j] for j in cItems1_idx]
            cVar0 = _get_cluster(mat_cov, cItems0)
            cVar1 = _get_cluster(mat_cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0_idx] *= alpha
            w[cItems1_idx] *= 1 - alpha

    return w


def _get_cluster(mat_cov: NDArray[np.float64], cItems: list[int]) -> float:
    """ Compute cluster for variance.

    Parameters
    ----------
    mat_cov: np.ndarray
        Covariance matrix.
    cItems: list or np.ndarray of int
        Cluster asset indices.

    Returns
    -------
    cVar: float
        Cluster variance

    """
    cov_ = mat_cov[np.ix_(cItems, cItems)]
    w_ = _get_IVP(cov_).reshape(-1, 1)
    cVar = (w_.T @ cov_) @ w_

    return float(cVar.item())


def _get_IVP(mat_cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Compute the inverse-variance matrix.

    Parameters
    ----------
    mat_cov : NDArray[np.float64]
        Variance-covariance matrix.

    Returns
    -------
    NDArray[np.float64]
        Inverse-variance weights.

    """
    ivp = 1. / np.diag(mat_cov)
    ivp /= np.sum(ivp)

    return ivp


def HRP(
    X: NDArray[np.float64],
    method: str = 'single',
    metric: str = 'euclidean',
    low_bound: float = 0.,
    up_bound: float = 1.0,
) -> NDArray[np.float64]:
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
        :math:`\leq w_i \leq` up_bound :math:`\forall i`. Default is 0 and 1.

    Returns
    -------
    np.ndarray
        Vector of weights computed by HRP algorithm.

    References
    ----------
    .. [2] https://ssrn.com/abstract=2708678

    """
    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    up_bound = max(up_bound, 1.0 / N)
    low_bound = min(low_bound, 1.0 / N)

    mat_cov = np.cov(X, rowvar=False)
    diag_cov = np.sqrt(np.diag(mat_cov))
    outer_diag = np.outer(diag_cov, diag_cov)
    with np.errstate(invalid='ignore', divide='ignore'):
        mat_corr = np.divide(mat_cov, outer_diag)
    mat_corr = np.clip(np.nan_to_num(mat_corr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    mat_dist = ((1.0 - mat_corr) / 2.0) ** 0.5

    mat_dist_upper = mat_dist[np.triu_indices(N, k=1)]
    link = sch.linkage(mat_dist_upper, method=method, metric=metric)
    sortIx = _get_quasi_diag(link)
    w_sorted = _get_rec_bisec(mat_cov, sortIx)

    w = np.empty(N)
    for i, col_idx in enumerate(sortIx):
        w[col_idx] = w_sorted[i]

    return _normalize(w.reshape([N, 1]), up_bound=up_bound, low_bound=low_bound)


# =========================================================================== #
#                        Inverse Variance Portfolio                           #
# =========================================================================== #


def IVP(
    X: NDArray[np.float64],
    normalize: bool = False,
    low_bound: float = 0.,
    up_bound: float = 1.0,
) -> NDArray[np.float64]:
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
        :math:`\leq w_i \leq` up_bound :math:`\forall i`. Default is 0 and 1.

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
#                         Minimum Variance Portfolio                          #
# =========================================================================== #


def MVP(
    X: NDArray[np.float64],
    normalize: bool = False,
) -> NDArray[np.float64]:
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
            print(mat_cov)
            raise np.linalg.LinAlgError

    e = np.ones([iv.shape[0], 1])
    w = (iv @ e) / (e.T @ iv @ e)

    if normalize:
        w = w - np.min(w)

        return w / np.sum(w)

    return w


def MVP_uc(
    X: NDArray[np.float64],
    w0: NDArray[np.float64] | None = None,
    up_bound: float = 1.,
    low_bound: float = 0.,
) -> NDArray[np.float64]:
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
        :math:`\leq w_i \leq` up_bound :math:`\forall i`. Default is 0 and 1.

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


def MDP(
    X: NDArray[np.float64],
    w0: NDArray[np.float64] | None = None,
    up_bound: float = 1.,
    low_bound: float = 0.,
) -> NDArray[np.float64]:
    r""" Get weights of Maximum Diversified Portfolio allocation.

    Notes
    -----
    Weights of Maximum Diversification Portfolio, as described by Y. Choueifaty
    and Y. Coignard [5]_, verify the following problem:

    .. math::

        w = \text{arg max } D(w) \\
        u.c. \begin{cases}w'e = 1 \\ 0 \leq w_i \leq 1 \\ \end{cases}

    Where :math:`D(w)` is the diversified ratio of portfolio weighted by `w`.

    Parameters
    ----------
    X : array_like
        Each column is a series of price or return's asset.
    w0 : array_like, optional
        Initial weights to maximize.
    up_bound, low_bound : float, optional
        Respectively maximum and minimum values of weights, such that low_bound
        :math:`\leq w_i \leq` up_bound :math:`\forall i`. Default is 0 and 1.

    Returns
    -------
    array_like
        Weights that maximize the diversified ratio of the portfolio.

    See Also
    --------
    diversified_ratio

    References
    ----------
    .. [5] `Choueifaty, Y., and Coignard, Y., 2008, Toward Maximum
           Diversification. <https://www.tobam.fr/wp-content/uploads/2014/12/
           TOBAM-JoPM-Maximum-Div-2008.pdf>`_

    """
    T, N = X.shape
    up_bound = max(up_bound, 1 / N)

    # Set function to minimze
    def f_max_divers_weights(w):
        return - diversified_ratio(X, W=w).flatten()

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


def rolling_allocation(
    f: Callable[..., NDArray[np.float64]],
    X: NDArray[np.float64],
    n: int = 252,
    s: int = 63,
    ret: bool = True,
    drift: bool = True,
    **kwargs,
) -> NDArray[np.float64]:
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
    X = pd.DataFrame(X).ffill()
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
        sub_X = sub_X.bfill()
        # Compute weights
        if len(assets) == 1:
            w = np.array([[1.]])

        else:
            w = f(sub_X.loc[:, assets].values, **kwargs)

        w_mat.loc[roll.d, assets] = w.flatten()
        w_mat.loc[roll.d, :] = w_mat.loc[roll.d, :].fillna(0.)
        # Compute portfolio performance
        perf = _perf_alloc(
            X.loc[slice_s, assets].bfill().values,
            w=w,
            drift=drift
        )
        portfolio.loc[slice_s] = portfolio.loc[roll.d] * perf.flatten()

    w_mat = w_mat.ffill().fillna(0.)

    return portfolio, w_mat


# =========================================================================== #
#                                   Tools                                     #
# =========================================================================== #


def _perf_alloc(X: NDArray[np.float64], w: NDArray[np.float64], drift: bool = True) -> NDArray[np.float64]:
    # Compute portfolio performance following specified weights
    if w.ndim == 1 and not isinstance(w, pd.Series):
        w = w.reshape([w.size, 1])

    if drift:
        return (X / X[0, :]) @ w

    perf = np.zeros(X.shape)
    perf[1:] = (X[1:] / X[:-1] - 1)

    return np.cumprod(perf @ w + 1)


def _normalize(w: NDArray[np.float64], low_bound: float = 0., up_bound: float = 1., sum_w: float = 1., max_iter: int = 1000) -> NDArray[np.float64]:
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
