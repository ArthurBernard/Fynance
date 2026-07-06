#!/usr/bin/env python3
# coding: utf-8

""" Tests for :mod:`fynance.core.checks` (protocol conformance + causality). """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.core import (
    Allocator,
    CostModel,
    DataSource,
    FeatureTransform,
    Metric,
    SignalModel,
)
from fynance.core.checks import assert_causal, check_conforms
from fynance.features.indicators import realized_volatility
from fynance.features.momentums import ema, sma

# =========================================================================== #
#                               assert_causal                                 #
# =========================================================================== #


def test_assert_causal_passes_on_real_trailing_features():
    for func in (sma, ema, realized_volatility):
        assert_causal(func, T=128, seed=0)


def test_assert_causal_raises_on_centered_mean():
    def centered_mean(x, w=9):
        return np.convolve(x, np.ones(w) / w, mode='same')

    with pytest.raises(AssertionError) as excinfo:
        assert_causal(centered_mean, T=64, seed=0)

    message = str(excinfo.value)
    assert 'centered_mean' in message
    assert 't0=' in message
    # earliest leaking index is reported and lies before the probe t0
    assert 'index' in message


def test_assert_causal_raises_on_global_zscore():
    def global_zscore(x):
        return (x - x.mean()) / x.std()

    with pytest.raises(AssertionError, match='global_zscore'):
        assert_causal(global_zscore, T=64, seed=0)


def test_assert_causal_nan_head_passes():
    def nan_head(x, k=5):
        out = np.array(x, dtype=float)
        out[:k] = np.nan
        return out

    assert_causal(nan_head, T=64, seed=0)


def test_assert_causal_atol_respected():
    def almost_causal(x, w=5, leak=1e-6):
        out = np.empty_like(x)
        for t in range(len(x)):
            lo = max(0, t - w + 1)
            out[t] = x[lo:t + 1].mean()

        return out + leak * x[-1]

    with pytest.raises(AssertionError):
        assert_causal(almost_causal, T=64, seed=0, atol=0.0)

    # the leak is tiny relative to the price scale: a generous atol absorbs it
    assert_causal(almost_causal, T=64, seed=0, atol=1e-2)


def test_assert_causal_single_t0():
    def trailing_mean(x, w=5):
        out = np.empty_like(x)
        for t in range(len(x)):
            lo = max(0, t - w + 1)
            out[t] = x[lo:t + 1].mean()

        return out

    assert_causal(trailing_mean, T=64, t0=32, seed=0)


def test_assert_causal_2d_output():
    def broadcast_trailing_mean(x, w=5):
        out = np.empty((len(x), 2))
        for t in range(len(x)):
            lo = max(0, t - w + 1)
            out[t] = x[lo:t + 1].mean()

        return out

    assert_causal(broadcast_trailing_mean, T=64, seed=0)


# =========================================================================== #
#                              check_conforms                                 #
# =========================================================================== #


class _FeatureTransform:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        return X - self.mean_


class _SignalModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class _Allocator:
    def __call__(self, data):
        n = data.shape[-1]
        return np.ones(n) / n


class _CostModel:
    def __call__(self, weights):
        return np.zeros(len(weights))


class _Metric:
    def __call__(self, returns, *args, **kwargs):
        return float(np.mean(returns))


def test_check_conforms_passes_on_minimal_conforming_objects():
    assert check_conforms(_FeatureTransform(), FeatureTransform) is None
    assert check_conforms(_SignalModel(), SignalModel) is None
    assert check_conforms(_Allocator(), Allocator) is None
    assert check_conforms(_CostModel(), CostModel) is None
    assert check_conforms(_Metric(), Metric) is None


def test_check_conforms_raises_on_predict_returning_list():
    class BadSignalModel(_SignalModel):
        def predict(self, X):
            return list(range(len(X)))

    with pytest.raises(AssertionError) as excinfo:
        check_conforms(BadSignalModel(), SignalModel)

    message = str(excinfo.value)
    assert 'predict' in message
    assert 'expected' in message


def test_check_conforms_raises_on_transform_wrong_shape():
    class BadFeatureTransform(_FeatureTransform):
        def transform(self, X):
            return X[:1]

    with pytest.raises(AssertionError) as excinfo:
        check_conforms(BadFeatureTransform(), FeatureTransform)

    message = str(excinfo.value)
    assert 'transform' in message
    assert 'expected' in message


def test_check_conforms_raises_on_metric_returning_array():
    class BadMetric(_Metric):
        def __call__(self, returns, *args, **kwargs):
            return returns

    with pytest.raises(AssertionError) as excinfo:
        check_conforms(BadMetric(), Metric)

    message = str(excinfo.value)
    assert '__call__' in message
    assert 'expected' in message


def test_check_conforms_datasource_raises_valueerror():
    class Src:
        def load(self, *args, **kwargs):
            return None

    with pytest.raises(ValueError, match='DataSource'):
        check_conforms(Src(), DataSource)


def test_check_conforms_unknown_protocol_raises_valueerror():
    class NotAProtocol:
        pass

    with pytest.raises(ValueError):
        check_conforms(_FeatureTransform(), NotAProtocol)
