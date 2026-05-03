""" Tests for portfolio allocation algorithms. """

import numpy as np
import pytest

from fynance.algorithms.allocation import ERC, HRP, IVP, MDP, MVP, MVP_uc

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
