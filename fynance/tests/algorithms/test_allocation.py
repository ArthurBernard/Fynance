""" Tests for portfolio allocation algorithms. """

import numpy as np
import pytest

from fynance.algorithms.allocation import ERC, HRP, IVP, MDP, MVP, MVP_uc, _normalize, _perf_alloc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
T, N = 200, 5


@pytest.fixture(scope="module")
def returns():
    """ Synthetic daily returns — 5 uncorrelated assets. """
    return RNG.normal(0.0, 0.01, size=(T, N))


@pytest.fixture(scope="module")
def correlated_returns():
    """ Returns with known positive covariance structure. """
    base = RNG.normal(0.0, 0.01, size=(T, 1))
    idio = RNG.normal(0.0, 0.005, size=(T, N))
    return base + idio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_weights(w, n=N, tol=1e-6):
    """ Weight vector is non-negative, sums to 1, correct shape. """
    w = np.asarray(w).flatten()
    assert w.shape == (n,), f"expected ({n},), got {w.shape}"
    assert np.all(w >= -tol), f"negative weight: {w.min():.6f}"
    assert abs(w.sum() - 1.0) < tol, f"weights sum to {w.sum():.6f}"


# ---------------------------------------------------------------------------
# IVP
# ---------------------------------------------------------------------------

def test_ivp_shape_and_sum(returns):
    w = IVP(returns)
    assert w.shape == (N, 1)
    assert abs(w.sum() - 1.0) < 1e-6


def test_ivp_positive(returns):
    w = IVP(returns).flatten()
    assert np.all(w >= 0)


def test_ivp_equal_variance():
    """ Equal-variance assets → equal weights. """
    X = RNG.normal(0.0, 0.01, size=(100, 4))
    # Scale all columns to same variance
    X = X / X.std(axis=0) * 0.01
    w = IVP(X).flatten()
    np.testing.assert_allclose(w, np.full(4, 0.25), atol=1e-4)


# ---------------------------------------------------------------------------
# MVP
# ---------------------------------------------------------------------------

def test_mvp_shape_and_sum(returns):
    w = MVP(returns)
    assert w.shape == (N, 1)
    assert abs(w.sum() - 1.0) < 1e-6


def test_mvp_normalized_positive(returns):
    w = MVP(returns, normalize=True).flatten()
    assert np.all(w >= -1e-8)
    assert abs(w.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# MVP_uc
# ---------------------------------------------------------------------------

def test_mvp_uc_shape_and_sum(returns):
    w = MVP_uc(returns)
    assert w.shape == (N, 1)
    assert abs(w.sum() - 1.0) < 1e-5


def test_mvp_uc_bounds(returns):
    w = MVP_uc(returns).flatten()
    assert np.all(w >= -1e-6)
    assert np.all(w <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# ERC
# ---------------------------------------------------------------------------

def test_erc_shape_and_sum(returns):
    w = ERC(returns)
    assert w.shape == (N, 1)
    assert abs(w.sum() - 1.0) < 1e-4


def test_erc_positive(returns):
    w = ERC(returns).flatten()
    assert np.all(w >= -1e-6)


def test_erc_equal_risk_uncorrelated():
    """ Uncorrelated equal-variance assets → near-equal weights. """
    X = np.column_stack([
        RNG.normal(0, 0.01, 300) for _ in range(4)
    ])
    w = ERC(X).flatten()
    np.testing.assert_allclose(w, np.full(4, 0.25), atol=0.05)


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------

def test_mdp_shape_and_sum(returns):
    w = MDP(returns)
    assert w.shape == (N, 1)
    assert abs(w.sum() - 1.0) < 1e-4


def test_mdp_positive(returns):
    w = MDP(returns).flatten()
    assert np.all(w >= -1e-6)


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------

def test_hrp_shape_and_sum(returns):
    w = HRP(returns)
    assert w.shape == (N, 1)
    assert abs(w.sum() - 1.0) < 1e-6


def test_hrp_positive(returns):
    w = HRP(returns).flatten()
    assert np.all(w >= -1e-6)


def test_hrp_correlated(correlated_returns):
    """ HRP should run without error on correlated assets. """
    w = HRP(correlated_returns)
    assert w.shape == (N, 1)
    assert abs(w.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# IVP normalize branch
# ---------------------------------------------------------------------------

def test_ivp_normalize(returns):
    w = IVP(returns, normalize=True).flatten()
    assert np.all(w >= -1e-8)
    assert abs(w.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _perf_alloc
# ---------------------------------------------------------------------------

def test_perf_alloc_drift_true():
    X = np.cumprod(1 + RNG.normal(0., 0.01, size=(50, 3)), axis=0)
    w = np.array([0.3, 0.3, 0.4])
    perf = _perf_alloc(X, w, drift=True)
    assert perf.shape[0] == 50


def test_perf_alloc_drift_false():
    X = np.cumprod(1 + RNG.normal(0., 0.01, size=(50, 3)), axis=0)
    w = np.array([0.3, 0.3, 0.4])
    perf = _perf_alloc(X, w, drift=False)
    assert perf.shape[0] == 50


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def test_normalize_raises_on_bad_bounds():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    with pytest.raises(ValueError):
        _normalize(w, low_bound=0.4, up_bound=0.9)


def test_normalize_clamps_weights():
    w = np.array([0.5, 0.3, 0.15, 0.05])
    w_norm = _normalize(w.copy(), low_bound=0.1, up_bound=0.4)
    assert np.all(w_norm >= 0.1 - 1e-9)
    assert np.all(w_norm <= 0.4 + 1e-9)


def test_normalize_max_iter_warning(capsys):
    w = np.array([0.99, 0.005, 0.003, 0.002])
    _normalize(w.copy(), low_bound=0.0, up_bound=0.3, max_iter=2)
    captured = capsys.readouterr()
    assert "exceeded max iterations" in captured.out
