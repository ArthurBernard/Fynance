#!/usr/bin/env python3
# coding: utf-8

""" Models conform to the SignalModel protocol (fit/predict). """

# Third-party packages
import numpy as np
import pytest
import torch
import torch.nn as nn

# Local packages
from fynance.core import SignalModel
from fynance.models import MultiLayerPerceptron
from fynance.models.gru import GatedRecurrentUnit
from fynance.models.lstm import LongShortTermMemory
from fynance.models.rnn import RecurrentNeuralNetwork


def _xy(n=40, feat=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, feat)).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True)).astype(np.float32)
    return X, y


# Recurrent models advertise the SignalModel protocol; they must honor
# fit(X, y) / predict(X) with a zero-initialized hidden (and cell) state.
RECURRENT_MODELS = [RecurrentNeuralNetwork, GatedRecurrentUnit, LongShortTermMemory]


@pytest.mark.parametrize("Model", RECURRENT_MODELS)
def test_recurrent_conforms_and_fit_predict(Model):
    X, y = _xy()
    model = Model(X, y, hidden_state_size=6)
    model.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)
    assert isinstance(model, SignalModel)
    # fit(X, y) must not raise and must return self; predict(X) returns Y only
    out = model.fit(X, y, epochs=2)
    assert out is model
    Y = out.predict(X)
    assert isinstance(Y, torch.Tensor)
    arr = np.asarray(Y)
    assert arr.shape == (X.shape[0], y.shape[1])


@pytest.mark.parametrize("Model", RECURRENT_MODELS)
def test_recurrent_fit_predict_float64_numpy(Model):
    """ Plain float64 numpy must fit/predict on a recurrent model. """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))  # float64
    y = X.sum(axis=1, keepdims=True)  # float64
    model = Model(X, y, hidden_state_size=5)
    model.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)
    out = model.fit(X, y, epochs=2).predict(X)
    assert np.asarray(out).shape[0] == X.shape[0]


def test_mlp_conforms_and_fit_predict():
    X, y = _xy()
    model = MultiLayerPerceptron(X, y, layers=[8])
    model.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)
    assert isinstance(model, SignalModel)
    out = model.fit(X, y, epochs=2).predict(X)
    arr = np.asarray(out)
    assert arr.shape[0] == X.shape[0]


def test_predict_accepts_numpy_and_tensor():
    X, y = _xy()
    model = MultiLayerPerceptron(X, y, layers=[4])
    model.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)
    model.fit(X, y, epochs=1)
    p_np = np.asarray(model.predict(X))
    p_t = np.asarray(model.predict(torch.as_tensor(X)))
    assert np.allclose(p_np, p_t, atol=1e-5)


def test_fit_returns_self():
    X, y = _xy()
    model = MultiLayerPerceptron(X, y, layers=[4])
    model.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)
    assert model.fit(X, y) is model


def test_fit_predict_on_float64_numpy():
    """ Plain float64 numpy (no .astype) must fit/predict without crashing. """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))  # float64 by default
    y = X.sum(axis=1, keepdims=True)  # float64
    assert X.dtype == np.float64 and y.dtype == np.float64
    model = MultiLayerPerceptron(X, y, layers=[8])
    model.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)
    out = model.fit(X, y, epochs=2).predict(X)
    assert np.asarray(out).shape[0] == X.shape[0]


def test_set_seed_only_seeds_requested_generator():
    """ Passing only seed_torch must leave the numpy seed untouched. """
    X, y = _xy()
    model = MultiLayerPerceptron(X, y, layers=[4])
    model.set_seed(seed_torch=123)
    assert model.seed_torch == 123
    assert model.seed_numpy is None  # numpy generator not (re)seeded
