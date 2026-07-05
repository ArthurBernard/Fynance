#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :mod:`fynance.research.importance`. """

# Third-party
import numpy as np
import pytest

# Local
from fynance.research import ImportanceResult, walk_forward_mda


class LinearModel:
    """ Closed-form OLS with intercept, via ``lstsq`` on ``[X, 1]``. """

    def fit(self, X, y):
        design = np.column_stack([X, np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.coef_ = coef[:-1]
        self.intercept_ = coef[-1]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class RecordingModel(LinearModel):
    """ Same closed-form OLS, but keeps a live reference *and* a snapshot of
    the train window at fit time so ``predict`` can assert the reference was
    never mutated by the permutation loop (which must only touch the test
    window's own copy).
    """

    def fit(self, X, y):
        self._train_ref = X
        self._train_snapshot = X.copy()
        return super().fit(X, y)

    def predict(self, X):
        assert np.array_equal(self._train_ref, self._train_snapshot), (
            "train window was mutated by the permutation loop"
        )
        return super().predict(X)


def _planted_signal(seed, T=600, k=4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, k))
    y = 2.0 * X[:, 0] + 0.1 * rng.standard_normal(T)
    return X, y


# -- planted signal ----------------------------------------------------------

def test_planted_signal_dominates_noise_features():
    X, y = _planted_signal(seed=1)

    result = walk_forward_mda(LinearModel, X, y, train=200, test=50, seed=0)

    assert isinstance(result, ImportanceResult)
    top = result.importances[0]
    for j in range(1, X.shape[1]):
        assert top > result.importances[j]
        assert abs(result.importances[j]) < 0.1 * top


# -- determinism ---------------------------------------------------------

def test_same_seed_gives_identical_arrays():
    X, y = _planted_signal(seed=2)

    r1 = walk_forward_mda(LinearModel, X, y, train=200, test=50, seed=7)
    r2 = walk_forward_mda(LinearModel, X, y, train=200, test=50, seed=7)

    np.testing.assert_array_equal(r1.importances, r2.importances)
    np.testing.assert_array_equal(r1.stds, r2.stds)
    assert r1.baseline == r2.baseline
    assert r1.n_folds == r2.n_folds


def test_different_seed_same_top_feature():
    X, y = _planted_signal(seed=3)

    r_a = walk_forward_mda(LinearModel, X, y, train=200, test=50, seed=1)
    r_b = walk_forward_mda(LinearModel, X, y, train=200, test=50, seed=2)

    assert np.argmax(r_a.importances) == 0
    assert np.argmax(r_b.importances) == 0
    assert not np.array_equal(r_a.importances, r_b.importances)


# -- permutation is test-window-only -----------------------------------------

def test_permutation_never_touches_train_window():
    X, y = _planted_signal(seed=4)

    # RecordingModel.predict raises AssertionError if the train window it
    # captured at fit time was ever mutated afterward; a plain, successful
    # call is the assertion that permutation stayed inside the test window.
    walk_forward_mda(RecordingModel, X, y, train=200, test=50, n_repeats=2,
                     seed=0)


# -- feature names -------------------------------------------------------

def test_feature_names_passthrough():
    X, y = _planted_signal(seed=5)
    names = ["signal", "n1", "n2", "n3"]

    result = walk_forward_mda(LinearModel, X, y, train=200, test=50,
                               feature_names=names)

    assert result.feature_names == names


def test_feature_names_length_mismatch_raises():
    X, y = _planted_signal(seed=5)

    with pytest.raises(ValueError, match="feature_names"):
        walk_forward_mda(LinearModel, X, y, train=200, test=50,
                         feature_names=["only_one"])


# -- validation ------------------------------------------------------------

def test_rejects_x_y_length_mismatch():
    X = np.zeros((100, 3))
    y = np.zeros(99)

    with pytest.raises(ValueError, match="length mismatch"):
        walk_forward_mda(LinearModel, X, y, train=50, test=10)


def test_rejects_non_2d_x():
    X = np.zeros(100)
    y = np.zeros(100)

    with pytest.raises(ValueError, match="2-D"):
        walk_forward_mda(LinearModel, X, y, train=50, test=10)


def test_rejects_train_plus_test_over_n():
    X = np.zeros((100, 3))
    y = np.zeros(100)

    with pytest.raises(ValueError, match="exceeds"):
        walk_forward_mda(LinearModel, X, y, train=80, test=30)


# -- purge smoke test ------------------------------------------------------

def test_purge_smoke():
    X, y = _planted_signal(seed=6, T=400)

    result = walk_forward_mda(LinearModel, X, y, train=150, test=40, purge=5,
                              seed=0)

    assert result.n_folds > 0
    assert result.importances.shape == (4,)
    assert result.stds.shape == (4,)
