#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :mod:`fynance.research.guards`. """

# Third-party
import numpy as np
import pytest

# Local
from fynance.research import (
    deflated_sharpe_ratio,
    gbm,
    permutation_test,
    probabilistic_sharpe_ratio,
)
from fynance.research.guards import _to_array  # noqa: F401 — coverage of helper
from fynance.strategy import Strategy


def momentum() -> Strategy:
    """ Causal sign-of-recent-change momentum. """
    return Strategy(features=lambda p: np.diff(p, prepend=p[0]))


# -- probabilistic / deflated Sharpe (deterministic) -----------------------

def test_psr_bounds_and_midpoint():
    assert probabilistic_sharpe_ratio(0.0, 100) == pytest.approx(0.5)
    val = probabilistic_sharpe_ratio(0.2, 250)
    assert 0.0 <= val <= 1.0


def test_psr_monotonic_in_sharpe_and_n():
    assert (probabilistic_sharpe_ratio(0.3, 250)
            > probabilistic_sharpe_ratio(0.1, 250))
    assert (probabilistic_sharpe_ratio(0.2, 1000)
            > probabilistic_sharpe_ratio(0.2, 100))


def test_dsr_equals_psr_at_one_trial():
    sr, n = 0.25, 500
    assert deflated_sharpe_ratio(sr, n, 1) == pytest.approx(
        probabilistic_sharpe_ratio(sr, n, sr_benchmark=0.0)
    )


def test_dsr_decreases_with_more_trials():
    sr, n = 0.25, 500
    few = deflated_sharpe_ratio(sr, n, 1)
    many = deflated_sharpe_ratio(sr, n, 100)
    assert many < few
    assert 0.0 <= many <= 1.0


# -- permutation test ------------------------------------------------------

def test_permutation_structure_and_determinism():
    out1 = permutation_test(momentum(), gbm(300, seed=1), n_permutations=30, seed=4)
    out2 = permutation_test(momentum(), gbm(300, seed=1), n_permutations=30, seed=4)

    assert set(out1) == {"observed", "p_value", "null_mean", "null_std"}
    assert 0.0 < out1["p_value"] <= 1.0
    assert out1 == out2  # fully reproducible


def test_permutation_detects_autocorrelation_edge():
    # AR(1) returns with positive autocorrelation -> momentum has a real edge;
    # shuffling destroys the autocorrelation, so the edge should be significant.
    rng = np.random.default_rng(0)
    n = 1200
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.35 * r[t - 1] + 0.01 * rng.standard_normal()
    prices = 100.0 * np.exp(np.cumsum(r))

    out = permutation_test(momentum(), prices, n_permutations=200, seed=0)

    assert out["observed"] > out["null_mean"]
    assert out["p_value"] < 0.10
