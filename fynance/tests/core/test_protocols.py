#!/usr/bin/env python3
# coding: utf-8

""" Tests for the :mod:`fynance.core.protocols` seams. """

# Third-party packages
import numpy as np

# Local packages
from fynance.core import (
    Allocator,
    CostModel,
    DataSource,
    FeatureTransform,
    Metric,
    SignalModel,
)


def test_feature_transform_runtime_checkable():
    class Dummy:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    assert isinstance(Dummy(), FeatureTransform)
    assert not isinstance(object(), FeatureTransform)


def test_signal_model_runtime_checkable():
    class Dummy:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert isinstance(Dummy(), SignalModel)
    assert not isinstance(object(), SignalModel)


def test_callable_protocols():
    class Alloc:
        def __call__(self, data):
            return np.ones(data.shape[-1])

    class Cost:
        def __call__(self, weights):
            return np.zeros(len(weights))

    class Sharpe:
        def __call__(self, returns, *a, **k):
            return float(np.mean(returns))

    assert isinstance(Alloc(), Allocator)
    assert isinstance(Cost(), CostModel)
    assert isinstance(Sharpe(), Metric)


def test_datasource_runtime_checkable():
    class Src:
        def load(self, *a, **k):
            return None

    assert isinstance(Src(), DataSource)
    assert not isinstance(object(), DataSource)
