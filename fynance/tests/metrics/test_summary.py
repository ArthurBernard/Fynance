#!/usr/bin/env python3
# coding: utf-8

""" Tests for the metrics summary and registry. """

# Third-party packages
import numpy as np

# Local packages
from fynance.core import Metric
from fynance.metrics import (
    METRICS,
    annual_volatility,
    sharpe,
    summary,
)


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


def test_summary_volatility_consistent_with_sharpe():
    # The reported annual_volatility must use the same log convention as the
    # reported sharpe (log=False), so that sharpe == annual_return / vol holds
    # on the displayed numbers (rf=0). Under the old default (log=True for vol,
    # log=False for sharpe) the displayed vol did not imply the displayed Sharpe.
    eq = np.array([100., 101.5, 99., 104., 108., 103., 110.])
    s = summary(eq)
    vol_log_false = float(annual_volatility(eq, period=252, log=False))
    assert np.isclose(s["annual_volatility"], vol_log_false)
    # sharpe (rf=0) implied by the displayed vol and return.
    assert np.isclose(s["sharpe"], s["annual_return"] / s["annual_volatility"])
