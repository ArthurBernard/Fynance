#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Regime-conditioned architecture: RegimeMoE conditions on the causal regime. """

# Third-party
import numpy as np
import pytest

# Local
from fynance.core import SignalModel
from fynance.metrics import sharpe
from fynance.models import ObjectiveModel, RegimeMoE


def _regime_edge_data(n=2000, block=100, seed=0):
    """ A *regime-dependent* edge: the profitable sign of feature ``s`` flips
    between two volatility regimes, so a model must know the regime to profit.

    Returns ``X = [level, s]`` (level is the positive price column the detector
    clusters on) and the realized returns.
    """
    rng = np.random.default_rng(seed)
    regime = (np.arange(n) // block) % 2          # 0,1 alternating blocks
    vol = np.where(regime == 0, 0.004, 0.03)      # distinct vol per regime
    level = 100 * np.exp(np.cumsum(rng.standard_normal(n) * vol))

    s = rng.choice([-1.0, 1.0], size=n)
    flip = np.where(regime == 0, 1.0, -1.0)       # edge sign flips by regime
    returns = (flip * s * 0.01 + rng.standard_normal(n) * 0.003).astype(np.float32)
    X = np.column_stack([level, s]).astype(np.float32)

    return X, returns


def test_conforms_to_signalmodel():
    assert isinstance(RegimeMoE(), SignalModel)


def test_predict_shape_and_bounds():
    X, y = _regime_edge_data(n=400, block=50)
    model = RegimeMoE(n_regimes=2, regime_w=10, epochs=10).fit(X, y)
    pos = np.asarray(model.predict(X))
    assert pos.shape == (400, 1)
    assert np.all(np.abs(pos) <= 1.0 + 1e-6)


def test_reproducible_with_seed():
    X, y = _regime_edge_data(n=600, block=50)
    a = RegimeMoE(n_regimes=2, regime_w=10, epochs=15, seed=7).fit(X, y).predict(X)
    b = RegimeMoE(n_regimes=2, regime_w=10, epochs=15, seed=7).fit(X, y).predict(X)
    assert np.allclose(a, b)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        RegimeMoE().predict(np.zeros((5, 2)))


def test_detector_fit_on_train_only_is_causal():
    # Labels on the train region must not change when more data is appended.
    X, y = _regime_edge_data(n=1200, block=100)
    model = RegimeMoE(n_regimes=2, regime_w=20).fit(X[:800], y[:800])
    train_labels = model.detector.predict(X[:800, 0])
    extended_labels = model.detector.predict(X[:, 0])
    assert np.array_equal(train_labels, extended_labels[:800])


def test_beats_regime_blind_model_on_regime_edge():
    # The edge is only exploitable if you know the regime: RegimeMoE should beat
    # a regime-blind ObjectiveModel given the exact same features.
    X, y = _regime_edge_data(n=2000, block=100)

    moe = RegimeMoE(
        n_regimes=2, regime_w=20, hidden=(16, 8), epochs=200, lr=5e-3, seed=0,
    ).fit(X, y)
    blind = ObjectiveModel(layers=(16, 8), epochs=200, lr=5e-3, seed=0).fit(X, y)

    moe_ret = np.asarray(moe.predict(X)).reshape(-1) * y
    blind_ret = np.asarray(blind.predict(X)).reshape(-1) * y

    moe_sharpe = sharpe(np.cumprod(1 + moe_ret), period=252)
    blind_sharpe = sharpe(np.cumprod(1 + blind_ret), period=252)

    assert moe_sharpe > blind_sharpe
    assert moe_sharpe > 1.0


def test_hard_routing_runs():
    X, y = _regime_edge_data(n=400, block=50)
    model = RegimeMoE(
        n_regimes=2, regime_w=10, routing='hard', epochs=10,
    ).fit(X, y)
    pos = np.asarray(model.predict(X))
    assert pos.shape == (400, 1)


def test_bad_routing_raises():
    X, y = _regime_edge_data(n=200, block=50)
    with pytest.raises(ValueError, match="routing"):
        RegimeMoE(routing='nope', regime_w=10).fit(X, y)
