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
