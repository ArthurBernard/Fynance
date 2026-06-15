""" Tests for portfolio allocation algorithms. """

import numpy as np
import pytest

from fynance.portfolio.allocation import (
    ERC,
    HRP,
    IVP,
    MDP,
    MVP,
    MVP_uc,
    _normalize,
    _perf_alloc,
    rolling_allocation,
)

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


def test_rolling_allocation_regression():
    """ rolling_allocation: pandas-free numpy output, golden regression.

    Values captured from the previous pandas implementation (parity
    verified across MVP/ERC/IVP/HRP) before the polars/numpy migration.
    """
    rng = np.random.RandomState(7)
    fac = rng.randn(130, 1)
    vols = np.linspace(0.005, 0.03, 4)
    prices = 100 * np.cumprod(1 + (0.5 * fac + rng.randn(130, 4)) * vols, axis=0)

    portfolio, w_mat = rolling_allocation(MVP, prices, n=50, s=15)

    # numpy outputs, not pandas
    assert isinstance(portfolio, np.ndarray)
    assert isinstance(w_mat, np.ndarray)
    assert portfolio.shape == (130,)
    assert w_mat.shape == (130, 4)
    # first n observations held at the initial value
    assert np.allclose(portfolio[:50], 100.)
    # each active weight row sums to 1 (MVP); 80 active rows here
    active = np.flatnonzero(np.abs(w_mat).sum(axis=1) > 1e-9)
    assert active.size == 80
    assert np.allclose(w_mat[active].sum(axis=1), 1.)
    # golden values
    assert np.allclose(portfolio[-3:],
                       [100.12393727, 99.78491062, 100.0663682])
    assert np.allclose(w_mat[-1],
                       [0.89803244, 0.09117386, 0.02558488, -0.01479118])
