#!/usr/bin/env python3
# coding: utf-8

""" Tests for the signal mappers and pipeline. """

# Third-party packages
import numpy as np

# Local packages
from fynance.core import SignalModel
from fynance.signal import (
    SignalPipeline,
    rank,
    sign,
    threshold,
    vol_target_position,
)


def test_sign():
    assert np.array_equal(sign(np.array([0.3, -0.1, 0.0])), [1.0, -1.0, 0.0])


def test_threshold_dead_band():
    out = threshold(np.array([0.5, 0.05, -0.5]), long=0.1, short=-0.1)
    assert np.array_equal(out, [1.0, 0.0, -1.0])


def test_rank_long_short_dollar_neutral():
    pred = np.array([[1.0, 2.0, 3.0, 4.0]])
    w = rank(pred, top=1, bottom=1)
    assert np.isclose(w.sum(), 0.0)          # dollar-neutral
    assert w[0, 3] == 1.0                     # highest -> long
    assert w[0, 0] == -1.0                    # lowest -> short
    assert w[0, 1] == 0.0 and w[0, 2] == 0.0


def test_rank_requires_2d():
    import pytest
    with pytest.raises(ValueError):
        rank(np.array([1.0, 2.0, 3.0]), top=1, bottom=1)


def test_vol_target_position_is_causal():
    rng = np.random.default_rng(0)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 100))
    signal = np.ones(100)
    pos = vol_target_position(signal, prices, w=10)
    assert pos.shape == (100,)
    # perturbing a future price must not change earlier positions
    p2 = prices.copy()
    p2[60:] *= 1.5
    pos2 = vol_target_position(signal, p2, w=10)
    assert np.allclose(pos[:60], pos2[:60], equal_nan=True)


def test_signal_pipeline():
    class DummyModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.asarray(X).reshape(-1) - 0.5

    pipe = SignalPipeline(DummyModel(), sign)
    pipe.fit(np.zeros((3, 1)), np.zeros(3))
    pos = pipe.predict_position(np.array([0.9, 0.1, 0.5]))
    assert np.array_equal(pos, [1.0, -1.0, 0.0])


def test_dummy_model_conforms_to_signalmodel():
    class DummyModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert isinstance(DummyModel(), SignalModel)
