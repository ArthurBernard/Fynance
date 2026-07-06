#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Distributional (quantile) regression: QuantileModel + PinballLoss. """

# Third-party
import numpy as np
import pytest
import torch

# Local
from fynance.core import SignalModel
from fynance.models import QuantileModel
from fynance.models.loss import PinballLoss


def _heteroskedastic_data(n=3000, seed=0):
    """ y = x + sigma(x) * noise: the spread of ``y`` grows with ``|x|``.

    A single-feature regression where the conditional quantiles are known
    analytically (``x + z_tau * sigma(x)`` for standard-normal noise), used
    to check empirical coverage of the fitted quantile band.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n).astype(np.float32)
    sigma = (0.1 + 0.4 * np.abs(x)).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    y = (x + sigma * noise).astype(np.float32)

    return x.reshape(-1, 1), y


# Fit once per module: training is the expensive part, and every test below
# only reads (never mutates) the fitted model's held-out predictions.
_TAUS = (0.1, 0.5, 0.9)
_X, _Y = _heteroskedastic_data()
_N_TRAIN = 2400
_X_TRAIN, _Y_TRAIN = _X[:_N_TRAIN], _Y[:_N_TRAIN]
_X_TEST, _Y_TEST = _X[_N_TRAIN:], _Y[_N_TRAIN:]


@pytest.fixture(scope="module")
def fitted_model():
    """ QuantileModel fit on the train slice only (no lookahead). """
    return QuantileModel(
        taus=_TAUS, layers=[16, 8], epochs=300, lr=1e-2, seed=0,
    ).fit(_X_TRAIN, _Y_TRAIN)


def test_conforms_to_signalmodel():
    assert isinstance(QuantileModel(), SignalModel)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        QuantileModel().predict(np.zeros((5, 1)))

    with pytest.raises(RuntimeError, match="fit"):
        QuantileModel().predict_quantiles(np.zeros((5, 1)))


def test_fit_returns_self():
    model = QuantileModel(epochs=2)
    assert model.fit(_X_TRAIN[:50], _Y_TRAIN[:50]) is model


def test_predict_quantiles_shape(fitted_model):
    q = fitted_model.predict_quantiles(_X_TEST)
    assert isinstance(q, np.ndarray)
    assert q.shape == (_X_TEST.shape[0], len(_TAUS))


def test_predict_shape_and_dtype(fitted_model):
    point = fitted_model.predict(_X_TEST)
    assert isinstance(point, np.ndarray)
    assert point.shape == (_X_TEST.shape[0],)


def test_predict_equals_median_column(fitted_model):
    # taus=(0.1, 0.5, 0.9) -> tau=0.5 is exactly present at column index 1.
    q = fitted_model.predict_quantiles(_X_TEST)
    point = fitted_model.predict(_X_TEST)
    assert np.array_equal(point, q[:, 1])


def test_nearest_to_half_when_median_absent():
    # taus without an exact 0.5: predict must pick the nearest column (0.4).
    model = QuantileModel(
        taus=(0.2, 0.4, 0.8), layers=[8], epochs=5, seed=0,
    ).fit(_X_TRAIN[:200], _Y_TRAIN[:200])
    q = model.predict_quantiles(_X_TEST[:20])
    assert np.array_equal(model.predict(_X_TEST[:20]), q[:, 1])  # tau=0.4


def test_quantiles_non_crossing_everywhere(fitted_model):
    q = fitted_model.predict_quantiles(_X_TEST)
    assert np.all(np.diff(q, axis=1) >= 0)


def test_taus_sorted_regardless_of_input_order():
    model = QuantileModel(taus=(0.9, 0.1, 0.5))
    assert model.taus == (0.1, 0.5, 0.9)


def test_held_out_coverage_in_expected_band(fitted_model):
    # The nominal [q10, q90] band should cover ~80% of held-out points; allow
    # a wide but meaningful tolerance around that for a small, quickly
    # trained net on a noisy heteroskedastic target.
    q = fitted_model.predict_quantiles(_X_TEST)
    lo, hi = q[:, 0], q[:, 2]
    coverage = np.mean((_Y_TEST >= lo) & (_Y_TEST <= hi))
    assert 0.72 <= coverage <= 0.88


def test_reproducible_with_seed():
    a = QuantileModel(taus=_TAUS, layers=[8], epochs=20, seed=7).fit(
        _X_TRAIN[:400], _Y_TRAIN[:400]).predict(_X_TEST[:50])
    b = QuantileModel(taus=_TAUS, layers=[8], epochs=20, seed=7).fit(
        _X_TRAIN[:400], _Y_TRAIN[:400]).predict(_X_TEST[:50])
    assert np.allclose(a, b)


def test_x_must_be_2d():
    with pytest.raises(ValueError):
        QuantileModel(epochs=2).fit(np.zeros(10), np.zeros(10))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        QuantileModel(epochs=2).fit(np.zeros((10, 1)), np.zeros(5))


def test_accepts_float64_numpy():
    # Plain float64 numpy (no .astype) must fit/predict without crashing.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))       # float64 by default
    y = X.sum(axis=1)                  # float64
    model = QuantileModel(layers=[8], epochs=5, seed=0).fit(X, y)
    out = model.predict(X)
    assert out.shape == (80,)


class TestPinballLossOnQuantileModel:
    def test_held_out_pinball_loss_is_finite(self, fitted_model):
        q = fitted_model.predict_quantiles(_X_TEST)
        loss = PinballLoss(taus=_TAUS)(
            torch.as_tensor(q), torch.as_tensor(_Y_TEST),
        )
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
