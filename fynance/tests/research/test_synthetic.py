#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :mod:`fynance.research.synthetic`. """

# Third-party
import numpy as np
import pytest

# Local
from fynance.core import PriceSeries
from fynance.features import detect_regimes
from fynance.research import gbm, regime_switching


@pytest.mark.parametrize("gen", [gbm, regime_switching])
def test_returns_price_series_of_right_length(gen):
    p = gen(500, seed=1)

    assert isinstance(p, PriceSeries)
    arr = p.to_numpy()
    assert arr.shape == (500,)
    assert arr.dtype == np.float64
    assert np.all(np.isfinite(arr)) and np.all(arr > 0)


@pytest.mark.parametrize("gen", [gbm, regime_switching])
def test_seed_determinism(gen):
    assert np.allclose(gen(300, seed=42).to_numpy(), gen(300, seed=42).to_numpy())
    assert not np.allclose(gen(300, seed=1).to_numpy(), gen(300, seed=2).to_numpy())


def test_gbm_statistics():
    mu, sigma, n = 0.0005, 0.02, 20000
    p = gbm(n, mu=mu, sigma=sigma, seed=7).to_numpy()
    log_ret = np.diff(np.log(p))

    assert log_ret.mean() == pytest.approx(mu, abs=5 * sigma / np.sqrt(n))
    assert log_ret.std() == pytest.approx(sigma, rel=0.1)


def test_regime_switching_feeds_detect_regimes():
    p = regime_switching(
        3000, regimes=((0.0, 0.005), (0.0, 0.04)), p_switch=0.01, seed=11
    )
    labels = detect_regimes(p.to_numpy(), n_regimes=2)

    assert len(np.unique(labels)) >= 2


def test_zero_length_edge_cases():
    # n == 1 -> a single starting price, no returns.
    assert gbm(1, s0=100.0, seed=0).to_numpy().tolist() == [100.0]
    assert regime_switching(1, s0=100.0, seed=0).to_numpy().tolist() == [100.0]


@pytest.mark.parametrize("gen", [gbm, regime_switching])
@pytest.mark.parametrize("n", [0, -1, -5])
def test_non_positive_length_raises(gen, n):
    # n < 1 used to silently return a length-1 path ([s0]) instead of erroring;
    # it must now raise rather than fabricate an observation out of no data.
    with pytest.raises(ValueError, match="positive integer"):
        gen(n, seed=0)


def test_regime_switching_initial_state_is_drawn():
    # With p_switch=0 the only regime randomness is the *initial* state. It used
    # to be hard-coded to regime 0 (biasing short paths); now it is drawn, so
    # across seeds both regimes must appear. Distinct sigmas make the active
    # regime visible in the first log-return's magnitude.
    regimes = ((0.0, 0.001), (0.0, 0.5))
    high_vol = []
    for s in range(40):
        p = regime_switching(2, regimes=regimes, p_switch=0.0, seed=s).to_numpy()
        first_ret = np.diff(np.log(p))[0]
        high_vol.append(abs(first_ret) > 0.05)  # True => started in regime 1

    assert any(high_vol)          # regime 1 was sometimes the initial state
    assert any(not h for h in high_vol)  # regime 0 too -> genuinely drawn


def test_regime_switching_initial_state_reproducible():
    a = regime_switching(50, p_switch=0.1, seed=9).to_numpy()
    b = regime_switching(50, p_switch=0.1, seed=9).to_numpy()

    assert np.allclose(a, b)
