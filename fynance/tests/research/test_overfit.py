#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :mod:`fynance.research.overfit` (CSCV / probability of backtest
overfitting).

"""

# Built-in
import math

# Third-party
import numpy as np
import pytest

# Local
from fynance.research import Experiment, gbm, pbo, returns_panel, run_experiment
from fynance.strategy import Strategy

# -- returns_panel -----------------------------------------------------------

def test_returns_panel_prefers_returns_series():
    a = Experiment(name="a", series={"returns": [0.01, -0.02, 0.03]})
    b = Experiment(name="b", series={"returns": [0.02, 0.0, -0.01]})

    panel = returns_panel([a, b])

    assert panel.shape == (3, 2)
    np.testing.assert_allclose(panel[:, 0], [0.01, -0.02, 0.03])
    np.testing.assert_allclose(panel[:, 1], [0.02, 0.0, -0.01])


def test_returns_panel_falls_back_to_equity_pct_change():
    a = Experiment(name="a", series={"equity": [100.0, 101.0, 99.0, 102.0]})

    panel = returns_panel([a])

    assert panel.shape == (3, 1)
    expected = np.array([101.0 / 100.0, 99.0 / 101.0, 102.0 / 99.0]) - 1.0
    np.testing.assert_allclose(panel[:, 0], expected)


def test_returns_panel_from_real_experiments():
    # Integration with the actual harness: run_experiment always populates both
    # 'equity' and 'returns', so a real Experiment feeds straight into the panel.
    strat = Strategy(features=lambda p: np.diff(p, prepend=p[0]))
    exps = [
        run_experiment(strat, gbm(300, seed=s), name=f"run-{s}", seed=s)
        for s in range(4)
    ]

    panel = returns_panel(exps)

    assert panel.shape == (299, 4)
    assert np.all(np.isfinite(panel))


def test_returns_panel_empty_raises():
    with pytest.raises(ValueError, match="at least one experiment"):
        returns_panel([])


def test_returns_panel_missing_series_raises():
    with pytest.raises(ValueError, match="neither a 'returns' nor an 'equity'"):
        returns_panel([Experiment(name="a")])


def test_returns_panel_short_equity_raises():
    with pytest.raises(ValueError, match="too short"):
        returns_panel([Experiment(name="a", series={"equity": [100.0]})])


def test_returns_panel_length_mismatch_raises():
    a = Experiment(name="a", series={"returns": [0.01, -0.02, 0.03]})
    b = Experiment(name="b", series={"returns": [0.01, -0.02]})

    with pytest.raises(ValueError, match="mismatched curve lengths"):
        returns_panel([a, b])


# -- pbo: shape / determinism -------------------------------------------------

def test_logits_length_matches_binomial_coefficient():
    rng = np.random.default_rng(0)
    panel = rng.normal(0.0, 0.01, size=(160, 6))

    result = pbo(panel, n_blocks=8)

    assert result.logits.shape == (math.comb(8, 4),)
    assert result.is_perf.shape == (math.comb(8, 4),)
    assert result.oos_perf.shape == (math.comb(8, 4),)


# -- pbo: statistical behaviour -----------------------------------------------

def test_pure_noise_panel_gives_pbo_near_half():
    # Pure noise, no config has a real edge: the IS winner should be an OOS
    # coin flip, so pbo should sit close to (but not required to equal) 0.5.
    rng = np.random.default_rng(0)
    panel = rng.normal(0.0, 0.01, size=(200, 20))

    result = pbo(panel, n_blocks=16)

    assert 0.35 <= result.pbo <= 0.65


def test_dominant_config_lowers_pbo_and_oos_loss():
    rng = np.random.default_rng(0)
    sigma = 0.01
    panel = rng.normal(0.0, sigma, size=(200, 20))
    noise_pbo = pbo(panel, n_blocks=16).pbo

    # A clearly dominant edge on one config: a per-period mean shift of 1
    # sigma is a large, unambiguous edge relative to the noise columns (whose
    # population mean is 0), so it should win in-sample AND hold up
    # out-of-sample almost every split.
    dominant = panel.copy()
    dominant[:, 0] += sigma
    result = pbo(dominant, n_blocks=16)

    assert result.pbo < noise_pbo
    assert result.prob_oos_loss < 0.05


def test_adversarial_construction_gives_high_pbo():
    # Block-alternating mean sign per config: each config performs well on
    # even blocks and poorly on odd blocks, or vice versa, depending on its
    # fixed rank. Whichever half of the blocks dominates the IS combination,
    # the complementary OOS half is dominated by the *other* parity almost
    # always (its blocks are the fixed-size complement of the IS blocks per
    # parity class) -- so the IS winner systematically becomes an OOS loser.
    seed = 42
    n_blocks, periods_per_block, n_configs = 16, 25, 12
    amplitude, sigma = 0.05, 0.01
    rank_signal = np.linspace(-1.0, 1.0, n_configs)

    rng = np.random.default_rng(seed)
    panel = np.empty((n_blocks * periods_per_block, n_configs))
    for b in range(n_blocks):
        parity = 1.0 if b % 2 == 0 else -1.0
        sl = slice(b * periods_per_block, (b + 1) * periods_per_block)
        panel[sl, :] = amplitude * parity * rank_signal[None, :]
    panel += rng.normal(0.0, sigma, size=panel.shape)

    result = pbo(panel, n_blocks=n_blocks)

    assert result.pbo > 0.7


def test_custom_metric_is_used():
    # Flipping the ranking direction (favor the lowest-mean config instead of
    # the highest) must change which config wins each split, and so the
    # resulting logits -- proof the callable actually drives the selection
    # rather than being ignored in favor of the default metric.
    rng = np.random.default_rng(3)
    panel = rng.normal(0.0, 0.01, size=(160, 6))

    calls = []

    def _spy_inverted_metric(returns):
        calls.append(returns.shape[0])
        return -float(np.mean(returns))

    default_result = pbo(panel, n_blocks=8)
    inverted_result = pbo(panel, n_blocks=8, metric=_spy_inverted_metric)

    # Called for both IS and OOS halves, for every config, every split.
    assert len(calls) == math.comb(8, 4) * 2 * 6
    assert not np.allclose(default_result.logits, inverted_result.logits)


# -- pbo: validation -----------------------------------------------------------

def test_odd_n_blocks_raises():
    panel = np.zeros((100, 5))
    with pytest.raises(ValueError, match="even"):
        pbo(panel, n_blocks=15)


def test_too_many_blocks_raises():
    panel = np.zeros((100, 5))
    with pytest.raises(ValueError, match="n_blocks > 16"):
        pbo(panel, n_blocks=18)


def test_n_blocks_of_16_is_allowed():
    rng = np.random.default_rng(0)
    panel = rng.normal(0.0, 0.01, size=(64, 4))
    result = pbo(panel, n_blocks=16)
    assert result.logits.shape == (math.comb(16, 8),)


def test_too_few_periods_raises():
    panel = np.zeros((10, 5))
    with pytest.raises(ValueError, match="fewer than n_blocks"):
        pbo(panel, n_blocks=16)


def test_1d_panel_raises():
    panel = np.zeros(100)
    with pytest.raises(ValueError, match="must be 2-D"):
        pbo(panel, n_blocks=8)
