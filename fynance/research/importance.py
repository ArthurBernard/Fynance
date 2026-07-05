#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Walk-forward permutation feature importance (MDA).

Mean-Decrease-Accuracy (MDA) importance evaluated **out-of-fold** on the
existing purged walk-forward splitter (:func:`fynance.data.split.walk_forward`):
for each fold the model is fit once on the train window, scored on the test
window, then re-scored after permuting one feature at a time *within the test
window only* -- the score drop attributed to that feature is averaged across
folds and repeats. Entry point: :func:`walk_forward_mda`.

"""

# Built-in
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local
from fynance.core import SignalModel
from fynance.data.split import walk_forward

__all__ = ['ImportanceResult', 'walk_forward_mda']


def _neg_mse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """ Negative mean squared error (higher is better). """
    return float(-np.mean((y_true - y_pred) ** 2))


@dataclass
class ImportanceResult:
    """ Result of :func:`walk_forward_mda`.

    Parameters
    ----------
    importances : numpy.ndarray, shape (n_features,)
        Mean out-of-sample score drop per permuted feature, averaged across
        folds and repeats. Higher means more important (permuting the feature
        hurts the score more).
    stds : numpy.ndarray, shape (n_features,)
        Standard deviation of the score drop across folds x repeats.
    baseline : float
        Mean (unpermuted) out-of-sample score across folds.
    n_folds : int
        Number of walk-forward folds evaluated.
    feature_names : list of str, optional
        Feature labels aligned with ``importances``/``stds``, if provided.

    """

    importances: NDArray[np.float64]
    stds: NDArray[np.float64]
    baseline: float
    n_folds: int
    feature_names: list[str] | None = None


def walk_forward_mda(
    model_factory: Callable[[], SignalModel],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    train: int = 252,
    test: int = 63,
    step: int | None = None,
    purge: int = 0,
    metric: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    n_repeats: int = 5,
    seed: int = 0,
    feature_names: list[str] | None = None,
) -> ImportanceResult:
    """ Mean-decrease-accuracy feature importance, evaluated out-of-fold.

    Splits ``X``/``y`` with the purged walk-forward generator
    :func:`fynance.data.split.walk_forward`. Per fold: fit ``model_factory()``
    once on the train window, score it on the (unpermuted) test window, then
    -- without refitting -- for each repeat and each feature, permute that
    feature's column **within the test window only** and re-score. The score
    drop (``baseline - permuted``) is the importance signal; it is averaged
    across folds x repeats (mean -> ``importances``, std -> ``stds``).

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a fresh model exposing
        ``fit(X, y) -> self`` / ``predict(X) -> ndarray``
        (:class:`fynance.core.SignalModel`). Called once per fold, so every
        fold starts from an untrained model.
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix, strictly time-ordered (row ``t`` must not depend on
        rows after ``t``).
    y : numpy.ndarray, shape (n_samples,)
        Target aligned with ``X``.
    train, test : int
        Train and test window lengths, forwarded to
        :func:`~fynance.data.split.walk_forward`.
    step : int, optional
        Roll step (defaults to ``test``; see
        :func:`~fynance.data.split.walk_forward`).
    purge : int
        Embargo between train and test windows, forwarded as-is.
    metric : callable, optional
        ``(y_true, y_pred) -> float``, **higher is better**. Defaults to the
        negative mean squared error (``-mean((y_true - y_pred) ** 2)``).
    n_repeats : int
        Number of independent permutation draws per feature per fold.
    seed : int
        Master seed. Fold ``i`` / repeat ``r`` draws its permutations from
        ``numpy.random.default_rng(seed + i * 1000 + r)``, so the result is
        fully reproducible for a fixed seed and independent across folds and
        repeats.
    feature_names : list of str, optional
        Labels carried through to :attr:`ImportanceResult.feature_names`;
        must match ``X.shape[1]`` if given.

    Returns
    -------
    ImportanceResult
        See :class:`ImportanceResult`.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, ``y`` is not 1-D, their lengths mismatch,
        ``train + test`` exceeds the number of samples (no fold would fit),
        or ``feature_names`` has the wrong length.

    Examples
    --------
    A closed-form linear model recovers the planted feature (column 0) as the
    most important, well above the pure-noise columns:

    >>> import numpy as np
    >>> from fynance.research.importance import walk_forward_mda
    >>> class LinearModel:
    ...     def fit(self, X, y):
    ...         design = np.column_stack([X, np.ones(len(X))])
    ...         self.coef_, *_ = np.linalg.lstsq(design, y, rcond=None)
    ...         return self
    ...     def predict(self, X):
    ...         design = np.column_stack([X, np.ones(len(X))])
    ...         return design @ self.coef_
    >>> rng = np.random.default_rng(0)
    >>> n, k = 300, 3
    >>> X = rng.standard_normal((n, k))
    >>> y = 2.0 * X[:, 0] + 0.1 * rng.standard_normal(n)
    >>> result = walk_forward_mda(LinearModel, X, y, train=150, test=50,
    ...                            n_repeats=3, seed=0)
    >>> result.n_folds
    3
    >>> bool(result.importances[0] > result.importances[1])
    True
    >>> bool(result.importances[0] > result.importances[2])
    True

    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got ndim={X.ndim}")

    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got ndim={y.ndim}")

    n_samples, n_features = X.shape

    if y.shape[0] != n_samples:
        raise ValueError(
            f"X and y length mismatch: X has {n_samples} rows, y has "
            f"{y.shape[0]}"
        )

    if train + test > n_samples:
        raise ValueError(
            f"train + test ({train + test}) exceeds the number of samples "
            f"({n_samples}); no walk-forward fold would fit"
        )

    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries, expected "
            f"{n_features} (X.shape[1])"
        )

    if metric is None:
        metric = _neg_mse

    fold_drops: list[NDArray[np.float64]] = []
    fold_baselines: list[float] = []

    windows = walk_forward(n_samples, train=train, test=test, step=step, purge=purge)
    for fold_idx, (train_idx, test_idx) in enumerate(windows):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])

        X_test = X[test_idx]
        y_test = y[test_idx]
        baseline_pred = model.predict(X_test)
        baseline_score = float(metric(y_test, baseline_pred))
        fold_baselines.append(baseline_score)

        repeat_drops = np.empty((n_repeats, n_features), dtype=np.float64)
        for r in range(n_repeats):
            rng = np.random.default_rng(seed + fold_idx * 1000 + r)
            for j in range(n_features):
                X_perm = X_test.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                permuted_score = float(metric(y_test, model.predict(X_perm)))
                repeat_drops[r, j] = baseline_score - permuted_score

        fold_drops.append(repeat_drops)

    all_drops = np.concatenate(fold_drops, axis=0)

    return ImportanceResult(
        importances=all_drops.mean(axis=0),
        stds=all_drops.std(axis=0),
        baseline=float(np.mean(fold_baselines)),
        n_folds=len(fold_drops),
        feature_names=feature_names,
    )
