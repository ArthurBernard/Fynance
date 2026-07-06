#!/usr/bin/env python3
# coding: utf-8

""" Purged walk-forward hyperparameter search.

Grid- or random-search over model hyperparameters, scored **out-of-sample**
on :func:`fynance.data.split.walk_forward` folds (no lookahead, optional
purge/embargo at each train/test boundary). This is the honest alternative to
picking hyperparameters on a single in-sample fit: every configuration is
scored on the same purged folds, and the number of configurations tried
(``n_trials``) is tracked so it can be fed straight into
:func:`fynance.research.guards.deflated_sharpe_ratio` to deflate the winning
configuration's Sharpe by the multiple-testing cost of the search itself.

"""

from __future__ import annotations

# Built-in packages
import itertools
from dataclasses import dataclass
from typing import Any, Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.data.split import walk_forward

__all__ = ['SearchResult', 'walk_forward_search']


def _neg_mse(y_true: NDArray, y_pred: NDArray) -> float:
    """ Default metric: negative mean squared error (higher is better). """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(-np.mean((y_true - y_pred) ** 2))


@dataclass
class SearchResult:
    """ Result of :func:`walk_forward_search`.

    Attributes
    ----------
    table : numpy.ndarray
        Out-of-sample scores, shape ``(n_configs, n_folds)`` -- row ``i``,
        column ``j`` is the score of ``params[i]`` on fold ``j``.
    params : list of dict
        Parameter dict tried for each row of ``table`` (same order).
    best_params : dict
        The configuration with the highest mean OOS score (mean over folds).
    best_model : Any
        ``factory(**best_params)`` refit on the **last** (most recent) train
        window -- the model to actually deploy.
    n_trials : int
        Number of configurations tried (``== len(params) == table.shape[0]``).
        Feed this into :func:`fynance.research.guards.deflated_sharpe_ratio`
        as ``n_trials`` so the winning configuration's Sharpe ratio is
        deflated by the number of configurations the search tried, rather
        than reported at face value.

    """

    table: NDArray
    params: list[dict[str, Any]]
    best_params: dict[str, Any]
    best_model: Any
    n_trials: int


def walk_forward_search(
    factory: Callable[..., Any],
    param_grid: dict[str, list[Any]],
    X: NDArray,
    y: NDArray,
    *,
    train: int = 252,
    test: int = 63,
    step: int | None = None,
    purge: int = 0,
    metric: Callable[[NDArray, NDArray], float] | None = None,
    n_iter: int | None = None,
    seed: int = 0,
) -> SearchResult:
    r""" Purged walk-forward hyperparameter search.

    Every configuration in ``param_grid`` is scored on the **same** purged
    walk-forward folds (:func:`fynance.data.split.walk_forward`), so
    configurations are compared honestly on out-of-sample data only -- no
    configuration ever sees a fold's test window during its own fit.

    Parameters
    ----------
    factory : callable
        ``factory(**params)`` must return a **fresh** model exposing
        ``fit(X, y)`` (returns the model, or ignored) and ``predict(X)``.
    param_grid : dict of list
        Maps parameter name to the list of candidate values. With ``n_iter``
        ``None`` (default) the **full grid** is tried, in the deterministic
        order of :func:`itertools.product` over ``param_grid.values()``
        (insertion order of ``param_grid``'s keys). With ``n_iter`` set, a
        size-``n_iter`` subsample of that same full grid is drawn without
        replacement, seeded by ``seed``.
    X, y : array_like
        Full, time-ordered feature matrix and target, both of length ``n``
        (the number of observations along axis 0).
    train, test : int
        Train and test window lengths forwarded to
        :func:`~fynance.data.split.walk_forward`. Default 252/63 (roughly one
        trading year / one quarter of daily bars).
    step : int, optional
        Roll step forwarded to :func:`~fynance.data.split.walk_forward`
        (defaults to ``test``, i.e. non-overlapping test windows).
    purge : int
        Embargo forwarded to :func:`~fynance.data.split.walk_forward`
        (observations dropped at the train/test boundary of every fold).
    metric : callable, optional
        ``metric(y_true, y_pred) -> float``, **higher is better**. Defaults
        to negative mean squared error (``-mean((y_true - y_pred) ** 2)``),
        so the default search maximizes prediction accuracy; pass a
        Sharpe-like metric to search directly for OOS risk-adjusted return.
    n_iter : int, optional
        If given, draw this many configurations at random (without
        replacement) from the full grid instead of trying every
        configuration; capped at the full grid size. ``None`` (default)
        tries the full grid.
    seed : int
        Seed for the ``n_iter`` random subsample. Ignored when ``n_iter`` is
        ``None``. The same ``seed`` always draws the same subsample.

    Returns
    -------
    SearchResult
        See :class:`SearchResult`.

    Raises
    ------
    ValueError
        If ``param_grid`` is empty or any of its value lists is empty (there
        would be nothing to search over), or if ``train + test`` exceeds the
        number of observations in ``X`` (no walk-forward fold would ever be
        produced, which would otherwise fail silently downstream with an
        empty ``table``).

    Examples
    --------
    A tiny closed-form ridge regression, searched over its regularization
    strength; the data is (near) noiseless and linear, so the unregularized
    fit (``alpha=0.0``) wins:

    >>> import numpy as np
    >>> from fynance.models.tuning import walk_forward_search
    >>> class RidgeModel:
    ...     ''' Closed-form ridge regression. '''
    ...     def __init__(self, alpha=0.0):
    ...         self.alpha = alpha
    ...     def fit(self, X, y):
    ...         n_features = X.shape[1]
    ...         gram = X.T @ X + self.alpha * np.eye(n_features)
    ...         self.coef_ = np.linalg.solve(gram, X.T @ y)
    ...         return self
    ...     def predict(self, X):
    ...         return X @ self.coef_
    >>> rng = np.random.default_rng(0)
    >>> T, F = 400, 2
    >>> X = rng.standard_normal((T, F))
    >>> w_true = np.array([1.5, -0.5])
    >>> y = X @ w_true + 0.01 * rng.standard_normal(T)
    >>> result = walk_forward_search(
    ...     RidgeModel, {"alpha": [0.0, 1.0, 10.0]}, X, y,
    ...     train=200, test=50, step=50,
    ... )
    >>> result.table.shape
    (3, 4)
    >>> result.n_trials
    3
    >>> result.best_params
    {'alpha': 0.0}
    >>> pred = result.best_model.predict(X[:3])
    >>> pred.shape
    (3,)

    The trial count plugs directly into the deflated Sharpe ratio, so the
    winning configuration's Sharpe can be reported honestly rather than at
    face value::

        from fynance.research.guards import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(sr, n_obs, result.n_trials)

    """
    if not param_grid or any(len(v) == 0 for v in param_grid.values()):
        raise ValueError(
            "param_grid must be a non-empty dict of non-empty lists, got "
            f"{param_grid!r}"
        )

    n = len(X)

    if train + test > n:
        raise ValueError(
            f"train + test ({train + test}) exceeds the number of "
            f"observations ({n}); no walk-forward fold would be produced"
        )

    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*(param_grid[k] for k in keys)))
    all_params = [dict(zip(keys, combo)) for combo in all_combos]

    if n_iter is None:
        params = all_params
    else:
        rng = np.random.default_rng(seed)
        n_draw = min(n_iter, len(all_params))
        idx = rng.choice(len(all_params), size=n_draw, replace=False)
        params = [all_params[i] for i in idx]

    if metric is None:
        metric = _neg_mse

    folds = list(walk_forward(n, train, test, step=step, purge=purge))
    n_folds = len(folds)

    table = np.empty((len(params), n_folds), dtype=np.float64)
    for i, p in enumerate(params):
        for j, (train_idx, test_idx) in enumerate(folds):
            model = factory(**p)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            table[i, j] = metric(y[test_idx], pred)

    mean_scores = table.mean(axis=1)
    best_i = int(np.argmax(mean_scores))
    best_params = params[best_i]

    last_train_idx, _ = folds[-1]
    best_model = factory(**best_params)
    best_model.fit(X[last_train_idx], y[last_train_idx])

    return SearchResult(
        table=table,
        params=params,
        best_params=best_params,
        best_model=best_model,
        n_trials=len(params),
    )
