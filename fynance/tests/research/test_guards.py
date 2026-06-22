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


def test_psr_matches_hand_computed_reference():
    # Worked reference (Bailey & Lopez de Prado PSR), hand-computed:
    #   sr = 0.10 (per-observation), n_obs = 100, benchmark = 0, normal returns
    #   denom = sqrt(1 - 0*sr + (3-1)/4 * sr^2) = sqrt(1 + 0.5 * 0.01) = 1.0024969
    #   z     = (0.10 - 0) * sqrt(100 - 1) / denom = 0.99250926
    #   PSR   = Phi(z) = 0.83952542
    assert probabilistic_sharpe_ratio(0.1, 100) == pytest.approx(
        0.8395254171844497, abs=1e-12
    )


def test_dsr_matches_hand_computed_reference():
    # Worked reference (Bailey & Lopez de Prado DSR), hand-computed:
    #   n_trials = 10, sr_variance = 0.04 (std 0.2)
    #   E[max] gauss factor = (1 - g) * Phi^-1(1 - 1/10)
    #                         + g * Phi^-1(1 - 1/(10 e)) = 1.57459830
    #     (g = Euler-Mascheroni = 0.5772156649)
    #   sr_star = sqrt(0.04) * 1.57459830 = 0.31491966
    #   then PSR(sr=0.30 per-obs, n_obs=500, benchmark=sr_star):
    #     denom = sqrt(1 + 0.5 * 0.30^2) = 1.0220568
    #     z     = (0.30 - 0.31491966) * sqrt(499) / denom = -0.32602512
    #     DSR   = Phi(z) = 0.37220268
    assert deflated_sharpe_ratio(0.3, 500, 10, sr_variance=0.04) == pytest.approx(
        0.3722026752956389, abs=1e-12
    )


def test_psr_dsr_not_saturated_for_realistic_inputs():
    # A realistic annualized Sharpe (~1.0 daily) de-annualized to per-obs
    # (1 / sqrt(252) = 0.063) must yield a sensible, non-saturated PSR/DSR.
    sr_obs = 1.0 / np.sqrt(252)
    psr = probabilistic_sharpe_ratio(sr_obs, 500)
    dsr = deflated_sharpe_ratio(sr_obs, 500, 10, sr_variance=0.04 / 252)

    assert 0.6 < psr < 0.999  # informative, not pinned at 1.0
    assert 0.3 < dsr < 0.95   # the multiple-testing correction still bites


# -- permutation test ------------------------------------------------------

def test_permutation_structure_and_determinism():
    out1 = permutation_test(momentum(), gbm(300, seed=1), n_permutations=30, seed=4)
    out2 = permutation_test(momentum(), gbm(300, seed=1), n_permutations=30, seed=4)

    assert set(out1) == {"observed", "p_value", "null_mean", "null_std"}
    assert 0.0 < out1["p_value"] <= 1.0
    assert out1 == out2  # fully reproducible


def stochastic_momentum() -> Strategy:
    """ Momentum whose features carry noise drawn from the global numpy RNG. """
    def features(p):
        base = np.diff(p, prepend=p[0])
        return base + 0.01 * np.std(base) * np.random.standard_normal(base.shape)

    return Strategy(features=features)


def test_permutation_threads_distinct_seeds_per_run():
    # Regression: the master seed used to be reused verbatim for the observed
    # run and every permutation, so a stochastic strategy replayed *identical*
    # model RNG draws on every path. The fix derives a distinct seed per run.
    # Reconstruct the old "reuse one seed" null and assert the real null is not
    # the same sequence, while staying fully reproducible.
    from fynance.research.runner import run_experiment

    strat = stochastic_momentum()
    prices = gbm(300, seed=1).to_numpy()
    log_ret = np.diff(np.log(prices))
    s0 = float(prices[0])
    seed, n_perm = 7, 30

    out = permutation_test(strat, prices, n_permutations=n_perm, seed=seed)

    # Old behavior: every run seeded with the same master seed.
    rng = np.random.default_rng(seed)
    legacy = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(log_ret)
        path = s0 * np.exp(np.concatenate([[0.0], np.cumsum(shuffled)]))
        legacy[i] = run_experiment(strat, path, name="perm", seed=seed
                                   ).metrics["sharpe"]

    # The fixed null differs from the legacy (one-seed) null for a stochastic
    # strategy: the model noise now varies across permutations too.
    assert not np.allclose([out["null_mean"]], [legacy.mean()])
    assert out["null_std"] > 0.0
    # Still reproducible for a fixed seed.
    out2 = permutation_test(strat, prices, n_permutations=n_perm, seed=seed)
    assert out == out2


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
