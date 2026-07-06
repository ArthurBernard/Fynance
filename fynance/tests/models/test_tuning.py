#!/usr/bin/env python3
# coding: utf-8

""" Tests for purged walk-forward hyperparameter search (models.tuning). """

# Built-in packages
import itertools

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.models.tuning import SearchResult, walk_forward_search


class RidgeModel:
    """ Closed-form ridge regression: coef = (X^T X + alpha I)^-1 X^T y. """

    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.coef_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        gram = X.T @ X + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(gram, X.T @ y)
        return self

    def predict(self, X):
        return X @ self.coef_


def _linear_data(n=800, n_features=3, noise=0.01, seed=42):
    """ Near-noiseless linear DGP: OLS (alpha=0) is the planted best config. """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    w_true = np.array([1.0, -2.0, 0.5])
    y = X @ w_true + noise * rng.standard_normal(n)
    return X, y


# ---------------------------------------------------------------------------
# Planted best config
# ---------------------------------------------------------------------------

def test_recovers_planted_best_config():
    X, y = _linear_data()
    grid = {"alpha": [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

    result = walk_forward_search(RidgeModel, grid, X, y, train=252, test=63)

    assert isinstance(result, SearchResult)
    assert result.best_params == {"alpha": 0.0}


def test_table_shape():
    X, y = _linear_data()
    grid = {"alpha": [0.0, 1.0, 10.0]}

    result = walk_forward_search(RidgeModel, grid, X, y, train=252, test=63)

    n_folds = 8  # (800 - 252) // 63 = 8 non-overlapping test windows
    assert result.table.shape == (3, n_folds)


def test_n_trials_matches_params():
    X, y = _linear_data()
    grid = {"alpha": [0.0, 1.0, 10.0, 100.0]}

    result = walk_forward_search(RidgeModel, grid, X, y, train=252, test=63)

    assert result.n_trials == len(result.params) == 4


def test_best_model_is_fitted_and_predicts():
    X, y = _linear_data()
    grid = {"alpha": [0.0, 1.0]}

    result = walk_forward_search(RidgeModel, grid, X, y, train=252, test=63)

    assert result.best_model.coef_ is not None
    pred = result.best_model.predict(X[:5])
    assert pred.shape == (5,)


# ---------------------------------------------------------------------------
# n_iter random subsampling
# ---------------------------------------------------------------------------

def test_n_iter_subsample_is_deterministic_per_seed():
    X, y = _linear_data()
    grid = {"alpha": [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

    r1 = walk_forward_search(
        RidgeModel, grid, X, y, train=252, test=63, n_iter=4, seed=7,
    )
    r2 = walk_forward_search(
        RidgeModel, grid, X, y, train=252, test=63, n_iter=4, seed=7,
    )

    assert r1.params == r2.params
    assert r1.n_trials == 4


def test_n_iter_subsample_is_subset_of_full_grid():
    X, y = _linear_data()
    grid = {"alpha": [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

    result = walk_forward_search(
        RidgeModel, grid, X, y, train=252, test=63, n_iter=3, seed=1,
    )

    full = [
        dict(zip(grid.keys(), combo))
        for combo in itertools.product(*grid.values())
    ]
    assert len(result.params) == 3
    assert all(p in full for p in result.params)


def test_n_iter_different_seed_can_change_subsample():
    X, y = _linear_data()
    grid = {"alpha": [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

    r1 = walk_forward_search(
        RidgeModel, grid, X, y, train=252, test=63, n_iter=3, seed=0,
    )
    r2 = walk_forward_search(
        RidgeModel, grid, X, y, train=252, test=63, n_iter=3, seed=123,
    )

    assert r1.params != r2.params


# ---------------------------------------------------------------------------
# ValueError paths
# ---------------------------------------------------------------------------

def test_rejects_empty_value_list():
    X, y = _linear_data()

    with pytest.raises(ValueError, match="param_grid"):
        walk_forward_search(RidgeModel, {"alpha": []}, X, y)


def test_rejects_empty_grid_dict():
    X, y = _linear_data()

    with pytest.raises(ValueError, match="param_grid"):
        walk_forward_search(RidgeModel, {}, X, y)


def test_rejects_train_plus_test_exceeding_n():
    X, y = _linear_data(n=100)

    with pytest.raises(ValueError, match="train \\+ test"):
        walk_forward_search(
            RidgeModel, {"alpha": [0.0]}, X, y, train=80, test=30,
        )
