#!/usr/bin/env python3
# coding: utf-8

""" Market-regime detection.

Unsupervised labelling of market regimes (e.g. calm / volatile, trending /
mean-reverting) by clustering rolling volatility and return features. Intended
for *analysis* and architecture conditioning (a model per regime, or a regime
embedding) — see the notes on causality.
"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray
from scipy.cluster.vq import kmeans2

# Local packages
from fynance.features.indicators import realized_volatility

__all__ = ['detect_regimes', 'regime_features', 'RegimeDetector']


def regime_features(X: NDArray, w: int = 21, period: int = 252) -> NDArray:
    """ Build the causal regime feature matrix: trailing vol and mean return.

    Both columns use **trailing** windows (past only), so the matrix is causal —
    the value at ``t`` depends on ``X[:t+1]`` only.

    Parameters
    ----------
    X : np.ndarray
        One-dimensional price/level series.
    w : int
        Rolling window.
    period : int
        Annualization factor for the volatility column.

    Returns
    -------
    np.ndarray
        Shape ``(len(X), 2)`` — ``[realized_volatility, rolling_mean_log_return]``.

    """
    X = np.asarray(X, dtype=np.float64).reshape(-1)
    vol = np.asarray(realized_volatility(X, w=w, period=period))
    ret = np.zeros_like(X)
    ret[1:] = np.log(X[1:] / X[:-1])
    roll_ret = np.zeros_like(X)
    for t in range(X.shape[0]):
        roll_ret[t] = ret[max(0, t - w + 1):t + 1].mean()

    return np.column_stack([vol, roll_ret])


class RegimeDetector:
    """ Causal market-regime detector (fit on the past, assign online).

    Unlike :func:`detect_regimes` (which clusters in-sample and therefore peeks
    at the whole series), this fits k-means on a **training** slice only and
    assigns any later point to the nearest training centroid — so labels are a
    **strictly causal** feature usable in a backtest. Labels are ordered by
    increasing mean volatility (0 = calmest), like :func:`detect_regimes`.

    Parameters
    ----------
    n_regimes : int
        Number of regimes (clusters).
    w : int
        Rolling window for the features.
    period : int
        Annualization factor for the volatility feature.
    seed : int
        Seed for the k-means initialization.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.features import RegimeDetector
    >>> rng = np.random.default_rng(0)
    >>> p = 100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01))
    >>> det = RegimeDetector(n_regimes=2, w=10).fit(p[:300])
    >>> labels = det.predict(p)
    >>> labels.shape, set(np.unique(labels)) <= {0, 1}
    ((400,), True)

    """

    def __init__(self, n_regimes: int = 3, w: int = 21, period: int = 252,
                 seed: int = 0):
        self.n_regimes = n_regimes
        self.w = w
        self.period = period
        self.seed = seed

    def fit(self, X: NDArray) -> RegimeDetector:
        """ Fit k-means on the training series ``X`` (past only). """
        feats = regime_features(X, self.w, self.period)
        self._mu = feats.mean(axis=0)
        self._sd = feats.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        z = (feats - self._mu) / self._sd

        centroids, labels = kmeans2(z, self.n_regimes, seed=self.seed,
                                    minit='++', missing='raise')
        self._centroids = centroids

        # Order clusters by mean (unstandardized) volatility for stable labels.
        vol = feats[:, 0]
        order = np.argsort([vol[labels == k].mean() for k in range(self.n_regimes)])
        remap = np.empty(self.n_regimes, dtype=int)
        remap[order] = np.arange(self.n_regimes)
        self._remap = remap

        return self

    def predict(self, X: NDArray) -> NDArray:
        """ Assign each point of ``X`` to the nearest training centroid. """
        feats = regime_features(X, self.w, self.period)
        z = (feats - self._mu) / self._sd
        dist = ((z[:, None, :] - self._centroids[None, :, :]) ** 2).sum(axis=-1)

        return self._remap[dist.argmin(axis=1)]

    def fit_predict(self, X: NDArray) -> NDArray:
        """ Convenience: :meth:`fit` then :meth:`predict` on the same series. """
        return self.fit(X).predict(X)


def detect_regimes(
    X: NDArray, n_regimes: int = 3, w: int = 21, period: int = 252, seed: int = 0,
) -> NDArray:
    r""" Label market regimes by k-means on rolling vol / return features.

    Builds two features per date — trailing realized volatility and trailing
    mean return — standardizes them, and clusters with k-means into
    ``n_regimes`` groups. Labels are **ordered by mean volatility** (0 = calmest,
    ``n_regimes - 1`` = most volatile) so they are comparable across runs.

    .. note::

       The clustering is fit **in-sample** (it sees the whole series), so the
       labels are appropriate for *analysis* and for studying regime
       conditioning — not as a strictly-causal online feature. A causal online
       assignment (fit on the past only) is a separate extension.

    Parameters
    ----------
    X : np.ndarray
        One-dimensional price/level series.
    n_regimes : int, optional
        Number of regimes (clusters). Default 3.
    w : int, optional
        Rolling window for the features. Default 21.
    period : int, optional
        Annualization factor for the volatility feature. Default 252.
    seed : int, optional
        Seed for the k-means initialization. Default 0.

    Returns
    -------
    np.ndarray
        Integer regime label per observation, shape ``(len(X),)``, in
        ``[0, n_regimes)`` and ordered by increasing average volatility.

    """
    feats = regime_features(X, w=w, period=period)
    vol = feats[:, 0]
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd == 0] = 1.0
    z = (feats - mu) / sd

    _, labels = kmeans2(z, n_regimes, seed=seed, minit='++', missing='raise')

    # Re-order labels by increasing mean volatility for stable interpretation.
    order = np.argsort([vol[labels == k].mean() for k in range(n_regimes)])
    remap = np.empty(n_regimes, dtype=int)
    remap[order] = np.arange(n_regimes)

    return remap[labels]
