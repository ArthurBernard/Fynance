#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Objective-aligned training: ObjectiveModel maximizes a financial loss. """

# Third-party
import numpy as np
import torch

# Local
from fynance.core import SignalModel
from fynance.metrics import sharpe
from fynance.models import ObjectiveModel, SortinoLoss


def _edge_data(n=1500, seed=0):
    """ A learnable edge: feature s in {-1,1} predicts the next return's sign. """
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=n)
    noise = rng.standard_normal(n) * 0.01
    returns = (s * 0.01 + noise).astype(np.float32)  # position s earns ~+1%/bar
    X = np.column_stack([s, rng.standard_normal(n)]).astype(np.float32)
    return X, returns


def test_conforms_to_signalmodel():
    assert isinstance(ObjectiveModel(), SignalModel)


def test_learns_a_known_edge():
    X, returns = _edge_data()
    model = ObjectiveModel(layers=(8,), epochs=150, lr=5e-3, seed=0).fit(X, returns)

    pos = np.asarray(model.predict(X)).reshape(-1)
    strat_ret = pos * returns
    # The learned positions should align with the edge -> clearly positive Sharpe.
    assert sharpe(np.cumprod(1 + strat_ret), period=252) > 1.0
    # and beat the do-nothing / wrong-way baseline.
    assert strat_ret.mean() > 0.0


def test_positions_are_bounded():
    X, returns = _edge_data(n=300)
    pos = np.asarray(ObjectiveModel(epochs=10).fit(X, returns).predict(X))
    assert pos.shape == (300, 1)
    assert np.all(np.abs(pos) <= 1.0 + 1e-6)


def test_reproducible_with_seed():
    X, returns = _edge_data(n=400)
    a = ObjectiveModel(epochs=20, seed=7).fit(X, returns).predict(X)
    b = ObjectiveModel(epochs=20, seed=7).fit(X, returns).predict(X)
    assert np.allclose(a, b)


def test_accepts_custom_net_and_loss():
    X, returns = _edge_data(n=300)
    net = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(),
                              torch.nn.Linear(4, 1))
    model = ObjectiveModel(net=net, loss=SortinoLoss(), epochs=10).fit(X, returns)
    assert np.asarray(model.predict(X)).shape == (300, 1)
