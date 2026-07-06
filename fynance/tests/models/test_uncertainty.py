#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predictive-uncertainty wrappers: DeepEnsemble and MCDropout. """

# Third-party
import numpy as np
import pytest
import torch
import torch.nn as nn

# Local
from fynance.core import SignalModel
from fynance.models.mlp import MultiLayerPerceptron
from fynance.models.uncertainty import DeepEnsemble, MCDropout

# --------------------------------------------------------------------------- #
# DeepEnsemble
# --------------------------------------------------------------------------- #

def _nonlinear_data(n=200, seed=0):
    """ A nonlinear single-feature regression target: y = sin(3x) + noise. """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1)).astype(np.float32)
    y = (np.sin(3 * X[:, 0]) + 0.05 * rng.standard_normal(n))
    return X, y.astype(np.float32).reshape(-1, 1)


class _TrainedMLP:
    """ SignalModel-conforming net with a baked-in epoch budget.

    ``BaseNeuralNet.fit`` defaults to a single full-batch step, which is too
    little to actually learn the nonlinear target used in these tests; this
    thin wrapper (composition, not inheritance) bakes the epoch/lr budget in so
    ``DeepEnsemble`` -- which only ever calls ``member.fit(X, y)`` -- still
    trains each member to a reasonable fit.
    """

    def __init__(self, n_in=1, hidden=16, epochs=60, lr=1e-2):
        self.net = MultiLayerPerceptron(n_in, 1, layers=[hidden],
                                        activation=nn.Tanh)
        self.net.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=lr)
        self.epochs = epochs

    def fit(self, X, y):
        self.net.fit(X, y, epochs=self.epochs)
        return self

    def predict(self, X):
        return self.net.predict(X)


def _factory(hidden=16, epochs=60, lr=1e-2):
    return lambda: _TrainedMLP(n_in=1, hidden=hidden, epochs=epochs, lr=lr)


def test_deepensemble_conforms_to_signalmodel():
    ens = DeepEnsemble(_factory(), n_members=3, seed=0)
    assert isinstance(ens, SignalModel)

    X, y = _nonlinear_data(n=60)
    out = ens.fit(X, y)
    assert out is ens

    pred = ens.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (60,)


def test_deepensemble_members_differ_and_std_positive():
    X, y = _nonlinear_data()
    ens = DeepEnsemble(_factory(), n_members=5, seed=0).fit(X, y)

    members = ens.predict_members(X)
    assert members.shape == (5, X.shape[0])

    for i in range(members.shape[0]):
        for j in range(i + 1, members.shape[0]):
            assert not np.allclose(members[i], members[j])

    assert np.all(ens.predict_std(X) > 0)


def test_deepensemble_deterministic_per_seed():
    X, y = _nonlinear_data()
    a = DeepEnsemble(_factory(), n_members=4, seed=3).fit(X, y).predict(X)
    b = DeepEnsemble(_factory(), n_members=4, seed=3).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_deepensemble_predict_is_mean_of_members():
    X, y = _nonlinear_data()
    ens = DeepEnsemble(_factory(), n_members=4, seed=1).fit(X, y)
    assert np.array_equal(ens.predict(X), ens.predict_members(X).mean(axis=0))


def test_deepensemble_predict_before_fit_raises():
    ens = DeepEnsemble(_factory(), n_members=2, seed=0)
    with pytest.raises(RuntimeError):
        ens.predict(np.zeros((5, 1), dtype=np.float32))


# --------------------------------------------------------------------------- #
# MCDropout
# --------------------------------------------------------------------------- #

def _dropout_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype(np.float32)
    y = np.sin(X[:, :1]).astype(np.float32)
    return X, y


def _dropout_net(hidden=16, drop=0.3):
    return MultiLayerPerceptron(2, 1, layers=[hidden], activation=nn.Tanh,
                                drop=drop)


def _dropout_free_net(hidden=16):
    return MultiLayerPerceptron(2, 1, layers=[hidden], activation=nn.Tanh)


def test_mcdropout_conforms_to_signalmodel():
    net = _dropout_net()
    net.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
    mc = MCDropout(net, n_samples=10, seed=0)
    assert isinstance(mc, SignalModel)

    X, y = _dropout_data(n=60)
    out = mc.fit(X, y)
    assert out is mc

    pred = mc.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (60,)


def test_mcdropout_fit_delegates_to_wrapped_model():
    net = _dropout_net()
    net.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
    X, y = _dropout_data(n=60)
    mc = MCDropout(net, n_samples=5, seed=0).fit(X, y)
    # the wrapped net trained (its optimizer has taken at least one step) and
    # is usable for prediction afterwards.
    assert mc.predict(X).shape == (60,)


def test_mcdropout_deterministic_per_seed():
    net = _dropout_net()
    net.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
    X, y = _dropout_data()
    net.fit(X, y, epochs=30)

    mc = MCDropout(net, n_samples=20, seed=7)
    a = mc.predict(X)
    b = mc.predict(X)
    assert np.array_equal(a, b)

    a_std = mc.predict_std(X)
    b_std = mc.predict_std(X)
    assert np.array_equal(a_std, b_std)


def test_mcdropout_std_positive_with_dropout():
    net = _dropout_net()
    net.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
    X, y = _dropout_data()
    net.fit(X, y, epochs=30)

    mc = MCDropout(net, n_samples=30, seed=0)
    assert np.all(mc.predict_std(X) > 0)


def test_mcdropout_mean_std_shrinks_with_more_samples():
    # Loose LLN check: Var(mean of n iid stochastic passes) ~ sigma^2 / n, so
    # the spread of the mean estimate across repeats (different seeds) should
    # shrink as n_samples grows from 10 to 200.
    net = _dropout_net()
    net.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
    X, y = _dropout_data()
    net.fit(X, y, epochs=30)

    def mean_pred(n_samples, seed):
        return MCDropout(net, n_samples=n_samples, seed=seed).predict(X).mean()

    n_repeats = 20
    means_10 = [mean_pred(10, s) for s in range(n_repeats)]
    means_200 = [mean_pred(200, s) for s in range(n_repeats)]

    assert np.std(means_200) < np.std(means_10)


def test_mcdropout_no_dropout_warns_and_std_near_zero():
    net = _dropout_free_net()
    net.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
    X, y = _dropout_data(n=80)
    net.fit(X, y, epochs=10)

    with pytest.warns(UserWarning):
        mc = MCDropout(net, n_samples=20, seed=0)

    assert np.allclose(mc.predict_std(X), 0.0, atol=1e-6)
