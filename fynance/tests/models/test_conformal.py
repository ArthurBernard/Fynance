#!/usr/bin/env python3
# coding: utf-8

""" Tests for causal split-conformal prediction (models.conformal). """

import numpy as np
import pytest
import torch

from fynance.models.conformal import (
    ConformalWrapper,
    _split_conformal_quantile,
    rolling_conformal,
)
from fynance.models.mlp import MultiLayerPerceptron


class LinearModel:
    """ Closed-form ordinary least squares -- no torch required. """

    def fit(self, X, y):
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        return X @ self.coef_


def _linear_data(n=3000, n_features=3, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    w_true = np.array([1.5, -0.5, 0.2])
    y = X @ w_true + noise * rng.standard_normal(n)
    return X, y


def _heteroskedastic_data(n=3000, seed=0):
    """ AR(1)-free heteroskedastic data: noise scale grows with |X[:, 0]|. """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    w_true = np.array([1.0, -1.0])
    scale = 0.2 + 2.0 * np.abs(X[:, 0])
    y = X @ w_true + scale * rng.standard_normal(n)
    return X, y


# --------------------------------------------------------------------------
# q_hat formula
# --------------------------------------------------------------------------

def test_q_hat_exact_formula_on_hand_built_residuals():
    # n=10, alpha=0.1 -> ceil(11 * 0.9) = ceil(9.9) = 10 -> the 10th of 10
    # sorted residuals, i.e. the maximum (1-indexed order statistic).
    residuals = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 8.0])
    q_hat = _split_conformal_quantile(residuals, alpha=0.1)
    assert q_hat == np.sort(residuals)[-1] == 9.0


def test_q_hat_smaller_alpha_can_hit_the_cap():
    # n=10, alpha=0.05 -> ceil(11 * 0.95) = ceil(10.45) = 11 > n=10, capped
    # at index n -> still the maximum residual (documented convention).
    residuals = np.arange(1, 11, dtype=np.float64)
    q_hat = _split_conformal_quantile(residuals, alpha=0.05)
    assert q_hat == residuals.max() == 10.0


def test_q_hat_mid_alpha_picks_expected_order_statistic():
    # n=20, alpha=0.5 -> ceil(21 * 0.5) = ceil(10.5) = 11 -> the 11th
    # smallest of 20 sorted residuals (1, ..., 20) is 11.
    residuals = np.arange(1, 21, dtype=np.float64)
    q_hat = _split_conformal_quantile(residuals, alpha=0.5)
    assert q_hat == 11.0


# --------------------------------------------------------------------------
# ConformalWrapper: validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_validation_raises(alpha):
    with pytest.raises(ValueError):
        ConformalWrapper(LinearModel(), alpha=alpha, window=10)


def test_window_not_smaller_than_T_raises_at_fit():
    X, y = _linear_data(n=50)
    wrapper = ConformalWrapper(LinearModel(), alpha=0.1, window=50)
    with pytest.raises(ValueError):
        wrapper.fit(X, y)


def test_window_larger_than_T_raises_at_fit():
    X, y = _linear_data(n=50)
    wrapper = ConformalWrapper(LinearModel(), alpha=0.1, window=100)
    with pytest.raises(ValueError):
        wrapper.fit(X, y)


def test_predict_interval_before_fit_raises():
    X, _ = _linear_data(n=50)
    wrapper = ConformalWrapper(LinearModel(), alpha=0.1, window=10)
    with pytest.raises(RuntimeError):
        wrapper.predict_interval(X)


# --------------------------------------------------------------------------
# ConformalWrapper: shapes and basic contract, with a closed-form model
# --------------------------------------------------------------------------

def test_fit_returns_self_and_sets_q_hat():
    X, y = _linear_data(n=500)
    wrapper = ConformalWrapper(LinearModel(), alpha=0.1, window=100)
    out = wrapper.fit(X, y)
    assert out is wrapper
    assert wrapper.q_hat_ is not None
    assert wrapper.q_hat_ > 0


def test_predict_shape():
    X, y = _linear_data(n=500)
    wrapper = ConformalWrapper(LinearModel(), alpha=0.1, window=100).fit(X, y)
    pred = wrapper.predict(X[-10:])
    assert pred.shape == (10,)


def test_predict_interval_shape_and_ordering():
    X, y = _linear_data(n=500)
    wrapper = ConformalWrapper(LinearModel(), alpha=0.1, window=100).fit(X, y)
    interval = wrapper.predict_interval(X[-10:])
    assert interval.shape == (10, 2)
    assert np.all(interval[:, 1] > interval[:, 0])
    pred = wrapper.predict(X[-10:])
    assert np.allclose(interval[:, 0], pred - wrapper.q_hat_)
    assert np.allclose(interval[:, 1], pred + wrapper.q_hat_)


def test_calibration_uses_only_trailing_window_not_seen_by_fit():
    # The wrapped model must never be fit on the calibration slice: swap
    # in a spy model that records exactly which rows it was trained on.
    X, y = _linear_data(n=500)
    window = 100

    class SpyModel(LinearModel):
        def fit(self, X, y):
            self.n_train_ = X.shape[0]
            return super().fit(X, y)

    spy = SpyModel()
    ConformalWrapper(spy, alpha=0.1, window=window).fit(X, y)
    assert spy.n_train_ == X.shape[0] - window


# --------------------------------------------------------------------------
# Marginal coverage on iid synthetic data
# --------------------------------------------------------------------------

def test_marginal_coverage_close_to_nominal_iid():
    X, y = _linear_data(n=3000, noise=1.0, seed=1)
    result = rolling_conformal(
        LinearModel, X, y, train=252, cal=63, test=63, alpha=0.1,
    )
    assert abs(result['coverage'] - 0.9) <= 0.05


def test_marginal_coverage_close_to_nominal_under_heteroskedasticity():
    # Constant-width intervals should still hit ~nominal *marginal*
    # coverage even though the data-generating noise scale varies bar to
    # bar -- that's the whole point (and limitation) of split-conformal:
    # coverage is honest on average, not conditionally per-regime.
    X, y = _heteroskedastic_data(n=3000, seed=2)
    result = rolling_conformal(
        LinearModel, X, y, train=252, cal=63, test=63, alpha=0.1,
    )
    assert abs(result['coverage'] - 0.9) <= 0.05


def test_constant_width_limitation_under_heteroskedasticity():
    # The interval half-width (q_hat) is a single scalar per window, so
    # every test bar in a window gets the *same* width regardless of the
    # local noise scale -- document this by asserting it directly.
    X, y = _heteroskedastic_data(n=3000, seed=2)
    result = rolling_conformal(
        LinearModel, X, y, train=252, cal=63, test=63, alpha=0.1,
    )
    widths = result['hi'] - result['lo']
    finite = widths[~np.isnan(widths)]
    # Within any single (non-overlapping) test window the width is constant.
    # Reconstruct window boundaries the same way rolling_conformal does.
    window, test = 252 + 63, 63
    t = 0
    n = len(y)
    while t + window + test <= n:
        block = widths[t + window:t + window + test]
        assert np.allclose(block, block[0])
        t += test
    # But the width clearly differs *across* windows/time (adapts to the
    # calibration window's residual scale) -- i.e. it is not one single
    # constant value for the whole series.
    assert len(np.unique(np.round(finite, 6))) > 1


# --------------------------------------------------------------------------
# rolling_conformal: validation, shapes, causality
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(train=0, cal=10, test=10),
    dict(train=10, cal=0, test=10),
    dict(train=10, cal=10, test=0),
    dict(train=-5, cal=10, test=10),
])
def test_rolling_conformal_window_validation_raises(kwargs):
    X, y = _linear_data(n=200)
    with pytest.raises(ValueError):
        rolling_conformal(LinearModel, X, y, **kwargs)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2])
def test_rolling_conformal_alpha_validation_raises(alpha):
    X, y = _linear_data(n=200)
    with pytest.raises(ValueError):
        rolling_conformal(LinearModel, X, y, train=50, cal=20, test=20, alpha=alpha)


def test_rolling_conformal_output_shapes_and_keys():
    X, y = _linear_data(n=600)
    result = rolling_conformal(LinearModel, X, y, train=200, cal=50, test=50)
    assert sorted(result.keys()) == ['coverage', 'covered', 'hi', 'lo', 'pred']
    for key in ('pred', 'lo', 'hi', 'covered'):
        assert result[key].shape == (600,)
    assert np.isnan(result['pred'][:250]).all()  # before first test window
    assert np.isfinite(result['pred'][250:300]).all()  # first test window


def test_rolling_conformal_coverage_is_nan_when_no_window_fits():
    X, y = _linear_data(n=50)
    result = rolling_conformal(LinearModel, X, y, train=30, cal=10, test=20)
    assert np.isnan(result['pred']).all()
    assert np.isnan(result['coverage'])


def test_rolling_conformal_causality_past_unchanged_by_future_perturbation():
    X, y = _linear_data(n=600, seed=3)
    kwargs = dict(train=200, cal=50, test=50, alpha=0.1)

    baseline = rolling_conformal(LinearModel, X.copy(), y.copy(), **kwargs)

    # Perturb X/y strictly after the first test window ends (bar 300+).
    X2, y2 = X.copy(), y.copy()
    perturb_from = 300
    X2[perturb_from:] += 100.0
    y2[perturb_from:] += 100.0

    perturbed = rolling_conformal(LinearModel, X2, y2, **kwargs)

    # Everything up to (and including) the first test window must be
    # bit-for-bit identical: it only ever depended on bars < perturb_from.
    first_window_end = 300
    for key in ('pred', 'lo', 'hi', 'covered'):
        np.testing.assert_array_equal(
            baseline[key][:first_window_end], perturbed[key][:first_window_end]
        )


def test_rolling_conformal_causality_perturbing_only_last_bar():
    X, y = _linear_data(n=600, seed=4)
    kwargs = dict(train=200, cal=50, test=50, alpha=0.1)

    baseline = rolling_conformal(LinearModel, X.copy(), y.copy(), **kwargs)

    X2, y2 = X.copy(), y.copy()
    X2[-1] += 50.0
    y2[-1] += 50.0
    perturbed = rolling_conformal(LinearModel, X2, y2, **kwargs)

    for key in ('pred', 'lo', 'hi', 'covered'):
        np.testing.assert_array_equal(baseline[key][:-1], perturbed[key][:-1])


# --------------------------------------------------------------------------
# Smoke test with a tiny torch model
# --------------------------------------------------------------------------

def test_smoke_conformal_wrapper_with_tiny_torch_mlp():
    rng = np.random.default_rng(5)
    T, N = 400, 3
    X = rng.standard_normal((T, N)).astype(np.float32)
    y = (X @ np.array([1.0, -1.0, 0.5], dtype=np.float32)
         + 0.1 * rng.standard_normal(T).astype(np.float32))
    y = y.reshape(-1, 1)

    model = MultiLayerPerceptron(N, 1, layers=[8])
    model.set_optimizer(torch.nn.MSELoss, torch.optim.Adam, lr=1e-2)

    class TorchAdapter:
        """ Thin adapter so BaseNeuralNet.fit's epochs kwarg is fixed. """

        def __init__(self, net):
            self.net = net

        def fit(self, X, y):
            self.net.fit(X, y, epochs=20)
            return self

        def predict(self, X):
            out = self.net.predict(X)
            return out.numpy() if hasattr(out, 'numpy') else np.asarray(out)

    wrapper = ConformalWrapper(TorchAdapter(model), alpha=0.1, window=80)
    wrapper.fit(X, y)
    assert wrapper.q_hat_ > 0

    interval = wrapper.predict_interval(X[-5:])
    assert interval.shape == (5, 2)
    assert np.all(interval[:, 1] > interval[:, 0])


def test_conformal_non_positive_window_raises():
    # Regression: window <= 0 must raise a clear ValueError at construction,
    # not crash later with an obscure IndexError on an empty residual array.
    with pytest.raises(ValueError, match="window"):
        ConformalWrapper(MultiLayerPerceptron, alpha=0.1, window=0)
    with pytest.raises(ValueError, match="window"):
        ConformalWrapper(MultiLayerPerceptron, alpha=0.1, window=-5)
