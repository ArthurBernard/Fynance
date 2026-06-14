#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.2 direction+magnitude stacking ensemble. """

import numpy as np
import torch
import torch.nn as nn

from fynance.models.ensemble import StackingEnsemble
from fynance.models.loss import DirectionalAccuracyLoss, SharpeLoss
from fynance.models.mlp import MultiLayerPerceptron

T, N_IN, N_OUT = 100, 4, 1


def _data():
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((T, N_IN)).astype(np.float32))
    y = torch.from_numpy(rng.standard_normal((T, N_OUT)).astype(np.float32))
    return X, y


def _ensemble():
    def direction():
        m = MultiLayerPerceptron(N_IN, N_OUT, layers=[8])
        m.set_optimizer(DirectionalAccuracyLoss, torch.optim.Adam, lr=1e-3)
        return m

    def magnitude():
        m = MultiLayerPerceptron(N_IN, N_OUT, layers=[8])
        m.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        return m

    def meta(n_features):
        m = MultiLayerPerceptron(n_features, N_OUT, layers=[4])
        m.set_optimizer(SharpeLoss, torch.optim.Adam, lr=1e-3)
        return m

    return StackingEnsemble(direction, magnitude, meta)


def test_fit_predict_shape():
    X, y = _data()
    out = _ensemble().fit_predict(X, y, train_period=40, test_period=10, roll_period=10)
    assert out.shape == (T, N_OUT)


def test_oof_nan_before_first_fold_and_finite_after():
    X, y = _data()
    out = _ensemble().fit_predict(X, y, train_period=40, test_period=10, roll_period=10)
    assert np.isnan(out[:40]).all()      # no OOF base features yet
    assert np.isfinite(out[50:]).all()   # predictions once folds exist


def test_meta_features_are_two_per_output():
    # The meta-model receives [direction_oof, magnitude_oof] -> 2 * N_OUT inputs.
    X, y = _data()
    captured = {}

    def meta(n_features):
        captured['n'] = n_features
        m = MultiLayerPerceptron(n_features, N_OUT, layers=[4])
        m.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        return m

    def base():
        m = MultiLayerPerceptron(N_IN, N_OUT, layers=[8])
        m.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        return m

    StackingEnsemble(base, base, meta).fit_predict(
        X, y, train_period=40, test_period=10, roll_period=10)
    assert captured['n'] == 2 * N_OUT
