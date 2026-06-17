#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" I1 — precomputed feature-matrix ``X`` / ``y`` threaded through the harness. """

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local
from fynance.models import MultiLayerPerceptron
from fynance.research import Experiment, gbm, run_experiment
from fynance.signal import sign
from fynance.strategy import Strategy


class LinearStub:
    """ Deterministic least-squares SignalModel stub (fit/predict). """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(X.shape[0], -1)
        self.w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w


def test_run_with_X_matches_features_callable():
    # Single-pass run: precomputed X must equal the same feature via a callable.
    prices = gbm(300, seed=1)
    feat = lambda p: np.sign(np.diff(p, prepend=p[0]))  # noqa: E731

    r_callable = Strategy(features=feat).run(prices)
    r_matrix = Strategy().run(prices, X=feat(prices.to_numpy()))

    assert np.allclose(r_callable.equity, r_matrix.equity)


def test_walk_forward_X_runs_and_uses_the_matrix():
    prices = gbm(400, seed=2)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(prices.to_numpy()), 2))
    y = (X[:, :1] - X[:, 1:2])  # a learnable linear target

    strat = Strategy(model=LinearStub(), signal=sign)
    res = strat.run_walk_forward(prices, y, train=80, test=40, X=X)

    assert np.all(np.isfinite(res.equity))
    # Positions actually depend on X: a different X gives a different result.
    res2 = strat.run_walk_forward(prices, y, train=80, test=40, X=rng.normal(size=X.shape))
    assert not np.allclose(res.equity, res2.equity)


def test_walk_forward_X_is_deterministic():
    prices = gbm(400, seed=2)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(prices.to_numpy()), 2))
    y = X[:, :1]

    a = Strategy(model=LinearStub()).run_walk_forward(prices, y, train=80, test=40, X=X)
    b = Strategy(model=LinearStub()).run_walk_forward(prices, y, train=80, test=40, X=X)
    assert np.allclose(a.equity, b.equity)


def test_no_lookahead_with_X_and_model():
    # Perturbing the price tail leaves the earlier OOS returns unchanged (X is
    # precomputed/causal; early windows never see the perturbed future).
    prices = gbm(600, seed=7).to_numpy()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(prices), 2))
    y = X[:, :1]
    wf = dict(train=120, test=60)

    base = Strategy(model=LinearStub()).run_walk_forward(prices, y, **wf, X=X)

    pert = prices.copy()
    cut = int(len(prices) * 0.7)
    pert[cut:] *= np.cumprod(1.0 + rng.standard_normal(len(prices) - cut) * 0.05)
    moved = Strategy(model=LinearStub()).run_walk_forward(pert, y, **wf, X=X)

    k = len(base.returns) // 3
    assert np.allclose(base.returns[:k], moved.returns[:k])


def test_run_experiment_threads_and_records_X():
    prices = gbm(400, seed=3)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(prices.to_numpy()), 2))
    y = X[:, :1]

    exp = run_experiment(Strategy(model=LinearStub(), signal=sign), prices,
                         name="xy", X=X, y=y, walk_forward=dict(train=80, test=40))

    assert isinstance(exp, Experiment)
    assert exp.spec["features"]["X_shape"] == [X.shape[0], X.shape[1]]
    assert exp.spec["features"]["names"] is None  # no names provided
    assert exp.spec["model"] == "LinearStub"      # model recorded in provenance
    assert np.all(np.isfinite(list(exp.metrics.values())))


def test_real_mlp_rolling_over_X():
    # The rolling-NN path: a real MLP refit per walk-forward window over X.
    prices = gbm(360, seed=5)
    rng = np.random.default_rng(0)
    n = len(prices.to_numpy())
    X = rng.normal(size=(n, 2)).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True)).astype(np.float32)

    mlp = MultiLayerPerceptron(2, 1, layers=[4])
    mlp.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)

    exp = run_experiment(Strategy(model=mlp, signal=np.sign), prices, name="mlp",
                         X=X, y=y, walk_forward=dict(train=120, test=60), seed=0)

    assert np.all(np.isfinite(list(exp.metrics.values())))
    assert exp.series["equity"]
