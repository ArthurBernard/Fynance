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

__all__ = ['detect_regimes']


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
    X = np.asarray(X, dtype=np.float64).reshape(-1)
    vol = np.asarray(realized_volatility(X, w=w, period=period))
    ret = np.zeros_like(X)
    ret[1:] = np.log(X[1:] / X[:-1])
    roll_ret = np.zeros_like(X)
    for t in range(X.shape[0]):
        roll_ret[t] = ret[max(0, t - w + 1):t + 1].mean()

    feats = np.column_stack([vol, roll_ret])
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd == 0] = 1.0
    feats = (feats - mu) / sd

    _, labels = kmeans2(feats, n_regimes, seed=seed, minit='++', missing='raise')

    # Re-order labels by increasing mean volatility for stable interpretation.
    order = np.argsort([vol[labels == k].mean() for k in range(n_regimes)])
    remap = np.empty(n_regimes, dtype=int)
    remap[order] = np.arange(n_regimes)

    return remap[labels]
