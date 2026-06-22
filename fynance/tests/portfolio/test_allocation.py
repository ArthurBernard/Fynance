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


def test_ivp_proportional_to_inverse_variance():
    """ Unequal-variance assets: weights ∝ 1/σ² (the documented formula). """
    rng = np.random.default_rng(1)
    vols = np.array([0.01, 0.04, 0.16])  # 1x / 4x / 16x
    X = rng.normal(0.0, 1.0, size=(4000, 3)) * vols
    var = np.diag(np.cov(X, rowvar=False))
    expected = (1.0 / var) / np.sum(1.0 / var)

    w = IVP(X).flatten()
    np.testing.assert_allclose(w, expected, rtol=1e-10)

    # normalize=True must preserve the inverse-variance ordering (no asset
    # zeroed by a subtract-min step) and keep a valid distribution.
    w_norm = IVP(X, normalize=True).flatten()
    assert np.all(w_norm > 0)
    assert abs(w_norm.sum() - 1.0) < 1e-9
    # With default box bounds the distribution is unchanged.
    np.testing.assert_allclose(w_norm, expected, rtol=1e-10)


def test_ivp_single_asset():
    """ N == 1 must not crash on the 0-d covariance and return [[1.]]. """
    X = RNG.normal(0.0, 0.01, size=(50, 1))
    w = IVP(X)
    assert w.shape == (1, 1)
    np.testing.assert_allclose(w, [[1.0]])


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


def test_mvp_single_asset():
    """ N == 1 must not crash on the 0-d covariance and return [[1.]]. """
    X = RNG.normal(0.0, 0.01, size=(50, 1))
    w = MVP(X)
    assert w.shape == (1, 1)
    np.testing.assert_allclose(w, [[1.0]])


def test_mvp_singular_covariance_pinv():
    """ An exactly singular covariance triggers the pseudo-inverse fallback.

    A constant (zero-variance) column makes its covariance row/column exactly
    zero, so the matrix is singular and ``np.linalg.inv`` raises
    ``LinAlgError`` — genuinely exercising the ``pinv`` branch (a merely
    near-singular matrix does not raise, so ``inv`` would silently succeed).
    """
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 0.01, size=(200, 1))
    b = rng.normal(0.0, 0.01, size=(200, 1))
    # Third column is constant → zero variance → exactly singular covariance.
    X = np.column_stack([a, b, np.full((200, 1), 100.0)])
    # Guard: this construction must actually make inv raise (else the test
    # would pass without ever reaching the pinv fallback it claims to cover).
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(np.atleast_2d(np.cov(X, rowvar=False)))
    w = MVP(X).flatten()
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6
    assert np.all(np.isfinite(w))


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


def test_mvp_uc_matches_closed_form_unequal_variance():
    """ On a long-only feasible set MVP_uc must match the closed-form MVP.

    Regression for the ftol/scaling bug: on small return-scale variances the
    objective w'Σw fell below SLSQP's default ftol and the optimizer returned
    the 1/N start instead of the minimum-variance weights.
    """
    rng = np.random.default_rng(0)
    vols = np.array([0.01, 0.04, 0.16])  # 1x / 4x / 16x, uncorrelated
    X = rng.normal(0.0, 1.0, size=(5000, 3)) * vols

    w_uc = MVP_uc(X).flatten()
    w_cf = MVP(X).flatten()
    # The closed-form solution here is long-only (positive), so it is feasible
    # for the box-constrained problem and the two must coincide.
    assert np.all(w_cf >= 0)
    np.testing.assert_allclose(w_uc, w_cf, atol=1e-4)
    # And it is clearly not the 1/N start (the old buggy output).
    assert not np.allclose(w_uc, np.full(3, 1 / 3), atol=1e-2)


def test_mvp_uc_single_asset():
    """ N == 1 must not crash on the 0-d covariance and return [[1.]]. """
    X = RNG.normal(0.0, 0.01, size=(50, 1))
    w = MVP_uc(X)
    assert w.shape == (1, 1)
    np.testing.assert_allclose(w, [[1.0]])


def test_mvp_uc_high_low_bound_still_sums_to_one(returns):
    """ low_bound > 1/N must be clamped, not break the sum-to-one constraint.

    With N=5 the feasible share per asset is 1/N=0.2; a low_bound of 0.3 makes
    the box incompatible with sum-to-one. Before the clamp, SLSQP silently
    returned weights summing to 1.5. low_bound is now clamped to 1/N (as
    HRP/IVP already do) so the weights still sum to one.
    """
    w = MVP_uc(returns, low_bound=0.3).flatten()
    assert abs(w.sum() - 1.0) < 1e-5, f"weights sum to {w.sum():.6f}"


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


def test_erc_equalizes_risk_contributions_unequal_variance():
    """ ERC must equalize risk contributions, not return the 1/N start.

    Regression for the ftol/scaling bug: the quartic risk-contribution
    surrogate on return-scale covariances was ~1e-16, below SLSQP's default
    ftol, so the optimizer stopped at iteration 1 and returned 1/N. On
    uncorrelated assets with vols 1x/4x/16x that gave risk contributions of
    roughly [0.006, 0.058, 0.936] instead of equalized thirds.
    """
    rng = np.random.default_rng(0)
    vols = np.array([0.01, 0.04, 0.16])  # 1x / 4x / 16x
    X = rng.normal(0.0, 1.0, size=(5000, 3)) * vols
    sigma = np.cov(X, rowvar=False)

    w = ERC(X).flatten()
    # Risk contributions RC_i = w_i (Σ w)_i, normalized to sum to one.
    rc = w * (sigma @ w)
    rc = rc / rc.sum()
    # Equal risk contributions: each asset carries 1/N of total risk.
    np.testing.assert_allclose(rc, np.full(3, 1 / 3), atol=1e-3)
    # The solution must not be the 1/N starting guess.
    assert not np.allclose(w, np.full(3, 1 / 3), atol=1e-2)
    # Lower-variance assets must carry more weight (ERC tilts toward them).
    assert w[0] > w[1] > w[2]


def test_erc_single_asset():
    """ N == 1 must not crash on the 0-d covariance and return [[1.]]. """
    X = RNG.normal(0.0, 0.01, size=(50, 1))
    w = ERC(X)
    assert w.shape == (1, 1)
    np.testing.assert_allclose(w, [[1.0]])


def test_erc_high_low_bound_still_sums_to_one(returns):
    """ low_bound > 1/N must be clamped, not break the sum-to-one constraint.

    With N=5 the feasible share per asset is 1/N=0.2; a low_bound of 0.3 makes
    the box incompatible with sum-to-one. Before the clamp, SLSQP silently
    returned weights summing to 1.5. low_bound is now clamped to 1/N (as
    HRP/IVP already do) so the weights still sum to one.
    """
    w = ERC(returns, low_bound=0.3).flatten()
    assert abs(w.sum() - 1.0) < 1e-4, f"weights sum to {w.sum():.6f}"


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


def test_mdp_beats_equal_weight_diversification():
    """ MDP weights must achieve a higher diversification ratio than 1/N. """
    from fynance.metrics import diversified_ratio

    rng = np.random.default_rng(3)
    # Two correlated blocks with different volatilities: 1/N is sub-optimal.
    f1 = rng.normal(0.0, 0.01, size=(600, 1))
    f2 = rng.normal(0.0, 0.01, size=(600, 1))
    idio = rng.normal(0.0, 0.003, size=(600, 4))
    vols = np.array([1.0, 1.0, 4.0, 4.0])
    X = np.column_stack([f1, f1, f2, f2]) * vols + idio

    w = MDP(X).flatten()
    dr_mdp = float(np.asarray(diversified_ratio(X, W=w)).item())
    dr_eq = float(np.asarray(diversified_ratio(X)).item())
    assert dr_mdp > dr_eq


def test_mdp_single_asset():
    """ N == 1 must not crash and return [[1.]]. """
    X = RNG.normal(0.0, 0.01, size=(50, 1))
    w = MDP(X)
    assert w.shape == (1, 1)
    np.testing.assert_allclose(w, [[1.0]])


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


def test_hrp_splits_risk_across_blocks():
    """ On two correlated blocks HRP must spread risk between the blocks.

    Block A (assets 0-1) is low-volatility, block B (assets 2-3) is high
    volatility. HRP's inverse-variance bisection should give block A more
    total weight than block B, yet keep both blocks meaningfully funded
    (unlike a pure minimum-variance solution that piles into the low-vol
    block).
    """
    rng = np.random.default_rng(5)
    fa = rng.normal(0.0, 0.01, size=(800, 1))
    fb = rng.normal(0.0, 0.01, size=(800, 1))
    idio = rng.normal(0.0, 0.002, size=(800, 4))
    vols = np.array([1.0, 1.0, 3.0, 3.0])
    X = np.column_stack([fa, fa, fb, fb]) * vols + idio

    w = HRP(X).flatten()
    block_a = w[0] + w[1]
    block_b = w[2] + w[3]
    # Low-vol block gets more weight than the high-vol block.
    assert block_a > block_b
    # But the high-vol block is still funded, not crushed to zero.
    assert block_b > 0.05
    # Within each block, the two near-identical assets are split evenly.
    np.testing.assert_allclose(w[0], w[1], rtol=0.2)
    np.testing.assert_allclose(w[2], w[3], rtol=0.2)


def test_hrp_single_asset():
    """ N == 1 must not crash on the 0-d covariance and return [[1.]]. """
    X = RNG.normal(0.0, 0.01, size=(50, 1))
    w = HRP(X)
    assert w.shape == (1, 1)
    np.testing.assert_allclose(w, [[1.0]])


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


def test_normalize_does_not_mutate_input():
    """ _normalize must copy its input, never mutate the caller's array. """
    w = np.array([0.5, 0.3, 0.15, 0.05])
    original = w.copy()
    out = _normalize(w, low_bound=0.1, up_bound=0.4)
    # Caller's array untouched.
    np.testing.assert_array_equal(w, original)
    # And a genuinely new array was returned.
    assert out is not w


def test_normalize_max_iter_warning():
    w = np.array([0.99, 0.005, 0.003, 0.002])
    with pytest.warns(UserWarning, match="exceeded max iterations"):
        _normalize(w.copy(), low_bound=0.0, up_bound=0.3, max_iter=2)


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
