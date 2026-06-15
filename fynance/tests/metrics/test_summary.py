#!/usr/bin/env python3
# coding: utf-8

""" Tests for the metrics summary and registry. """

# Third-party packages
import numpy as np

# Local packages
from fynance.core import Metric
from fynance.metrics import METRICS, sharpe, summary


def test_summary_keys():
    eq = np.array([100., 101., 103., 102., 105., 107.])
    s = summary(eq)
    assert set(s) == {"annual_return", "annual_volatility", "sharpe",
                      "sortino", "calmar", "max_drawdown"}
    for v in s.values():
        assert np.isfinite(v)


def test_summary_matches_direct_calls():
    eq = np.array([100., 101., 103., 102., 105., 107.])
    s = summary(eq)
    assert np.isclose(s["sharpe"], float(sharpe(eq)))


def test_registry_contains_metrics():
    assert "sharpe" in METRICS and "max_drawdown" in METRICS
    eq = np.array([100., 101., 103., 102., 105., 107.])
    assert np.isclose(float(METRICS["sharpe"](eq)), float(sharpe(eq)))


def test_metric_functions_conform_to_protocol():
    # plain callables structurally satisfy the Metric protocol
    assert isinstance(sharpe, Metric)
