""" Tests for risk decomposition (fynance.portfolio.attribution). """

import warnings

import numpy as np
import pytest

from fynance.portfolio.allocation import ERC
from fynance.portfolio.attribution import (
    marginal_risk,
    risk_contribution,
    roll_risk_contribution,
)
from fynance.portfolio.covariance import ledoit_wolf

# =========================================================================== #
#                          Basic invariants                                   #
# =========================================================================== #


class TestMarginalRisk:
    """ Tests for marginal_risk function. """

    def test_marginal_risk_basic(self):
        """ Marginal risk should be (sigma @ w) / sigma_p. """
        w = np.array([0.6, 0.4])
        sigma = np.array([[0.04, 0.0], [0.0, 0.01]])

        mr = marginal_risk(w, sigma)

        # sigma @ w = [0.04*0.6, 0.01*0.4] = [0.024, 0.004]
        # sigma_p = sqrt(0.024*0.6 + 0.004*0.4) = sqrt(0.01440 + 0.00160) = sqrt(0.016) = 0.1265
        # mr = [0.024/0.1265, 0.004/0.1265]
        expected = np.array([0.024, 0.004]) / np.sqrt(0.016)
        assert np.allclose(mr, expected)

    def test_marginal_risk_flattens_input(self):
        """ marginal_risk should accept (N, 1) and flatten it. """
        w = np.array([[0.5], [0.5]])
        sigma = np.array([[0.04, 0.0], [0.0, 0.01]])

        mr = marginal_risk(w, sigma)

        # Should produce same result as 1-D input
        mr_flat = marginal_risk(np.array([0.5, 0.5]), sigma)
        assert np.allclose(mr, mr_flat)

    def test_marginal_risk_zero_sigma_p(self):
        """ marginal_risk should return zeros when sigma_p == 0. """
        w = np.array([0.5, 0.5])
        sigma = np.zeros((2, 2))

        mr = marginal_risk(w, sigma)

        assert np.allclose(mr, 0.0)

    def test_marginal_risk_validates_square_sigma(self):
        """ marginal_risk should raise on non-square sigma. """
        w = np.array([0.5, 0.5])
        sigma = np.ones((2, 3))

        with pytest.raises(ValueError, match="square"):
            marginal_risk(w, sigma)

    def test_marginal_risk_validates_shape_match(self):
        """ marginal_risk should raise on shape mismatch. """
        w = np.array([0.5, 0.5, 0.5])
        sigma = np.eye(2)

        with pytest.raises(ValueError, match="does not match"):
            marginal_risk(w, sigma)


class TestRiskContribution:
    """ Tests for risk_contribution function. """

    def test_risk_contribution_pct_hand_checked(self):
        """ risk_contribution on 2-asset example: w=[0.5, 0.5], sigma=[[0.04,0],[0,0.01]]. """
        w = np.array([0.5, 0.5])
        sigma = np.array([[0.04, 0.0], [0.0, 0.01]])

        rc_pct = risk_contribution(w, sigma, pct=True)

        # Expected: [0.8, 0.2] (from contract specification)
        assert np.allclose(rc_pct, [0.8, 0.2])

    def test_risk_contribution_pct_sum_to_one(self):
        """ risk_contribution(pct=True) should sum to 1. """
        rng = np.random.default_rng(42)
        N = 6

        # Create a random PSD covariance matrix: A @ A.T
        A = rng.standard_normal((N, N))
        sigma = A @ A.T
        w = rng.uniform(0.1, 1.0, N)
        w /= w.sum()

        rc_pct = risk_contribution(w, sigma, pct=True)

        assert np.allclose(rc_pct.sum(), 1.0, rtol=1e-12)

    def test_risk_contribution_absolute_sum_to_sigma_p(self):
        """ risk_contribution(pct=False) should sum to sigma_p. """
        rng = np.random.default_rng(43)
        N = 6

        A = rng.standard_normal((N, N))
        sigma = A @ A.T
        w = rng.uniform(0.1, 1.0, N)
        w /= w.sum()

        rc_abs = risk_contribution(w, sigma, pct=False)
        sigma_p = np.sqrt(w @ sigma @ w)

        assert np.allclose(rc_abs.sum(), sigma_p, rtol=1e-12)

    def test_risk_contribution_homogeneity(self):
        """ risk_contribution(2*w, pct=True) should equal the w version. """
        rng = np.random.default_rng(44)
        N = 5

        A = rng.standard_normal((N, N))
        sigma = A @ A.T
        w = rng.uniform(0.1, 1.0, N)
        w /= w.sum()

        rc_w = risk_contribution(w, sigma, pct=True)
        rc_2w = risk_contribution(2.0 * w, sigma, pct=True)

        assert np.allclose(rc_w, rc_2w)

    def test_risk_contribution_zero_sigma_p(self):
        """ risk_contribution should return zeros when sigma_p == 0. """
        w = np.array([0.5, 0.5])
        sigma = np.zeros((2, 2))

        rc_pct = risk_contribution(w, sigma, pct=True)
        rc_abs = risk_contribution(w, sigma, pct=False)

        assert np.allclose(rc_pct, 0.0)
        assert np.allclose(rc_abs, 0.0)

    def test_risk_contribution_degenerate_no_warnings(self):
        """ risk_contribution on zero sigma should not raise warnings. """
        w = np.array([0.5, 0.5])
        sigma = np.zeros((2, 2))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rc = risk_contribution(w, sigma, pct=True)
            assert np.allclose(rc, 0.0)


# =========================================================================== #
#                              ERC integration                                #
# =========================================================================== #


class TestErcIntegration:
    """ Test risk contributions on ERC-allocated portfolios. """

    def test_erc_contributions_are_equal(self):
        """ risk_contribution of ERC weights should be approximately 1/N. """
        rng = np.random.default_rng(45)
        N, T = 5, 300
        X = rng.standard_normal((T, N))

        # Compute ERC weights
        w_erc = ERC(X).flatten()

        # Compute covariance
        sigma = np.cov(X, rowvar=False)

        # Compute risk contributions
        rc = risk_contribution(w_erc, sigma, pct=True)

        # All entries should be close to 1/N
        expected = 1.0 / N
        assert np.allclose(rc, expected, atol=1e-3)


# =========================================================================== #
#                           Diagonal sigma                                    #
# =========================================================================== #


class TestDiagonalSigma:
    """ Test on diagonal covariance matrices. """

    def test_diagonal_sigma_contributions(self):
        """ On diagonal sigma, contributions are proportional to w_i^2 * var_i. """
        # w = [0.3, 0.7], diag(sigma) = [0.04, 0.09]
        w = np.array([0.3, 0.7])
        sigma = np.diag([0.04, 0.09])

        rc_pct = risk_contribution(w, sigma, pct=True)

        # sigma_p^2 = w @ sigma @ w = 0.3^2 * 0.04 + 0.7^2 * 0.09 = 0.0477
        # rc_abs = w * (sigma @ w) = [0.3, 0.7] * [0.012, 0.063] = [0.0036, 0.0441]
        # rc_pct = rc_abs / sigma_p^2
        sigma_p_sq = 0.3**2 * 0.04 + 0.7**2 * 0.09
        expected = np.array([0.0036, 0.0441]) / sigma_p_sq

        assert np.allclose(rc_pct, expected, rtol=1e-10)


# =========================================================================== #
#                            Rolling risk contribution                        #
# =========================================================================== #


class TestRollRiskContribution:
    """ Tests for roll_risk_contribution function. """

    def test_roll_output_shape(self):
        """ roll_risk_contribution output should be (T, N). """
        rng = np.random.default_rng(46)
        T, N = 100, 5
        X = rng.standard_normal((T, N))
        W = rng.uniform(0.1, 1.0, (T, N))
        W /= W.sum(axis=1, keepdims=True)

        rc = roll_risk_contribution(W, X, n=20)

        assert rc.shape == (T, N)

    def test_roll_early_rows_are_nan(self):
        """ Rows t < n should be filled with NaN. """
        rng = np.random.default_rng(47)
        T, N = 100, 5
        n = 30
        X = rng.standard_normal((T, N))
        W = rng.uniform(0.1, 1.0, (T, N))
        W /= W.sum(axis=1, keepdims=True)

        rc = roll_risk_contribution(W, X, n=n)

        assert np.isnan(rc[:n]).all()
        assert np.any(~np.isnan(rc[n:]))

    def test_roll_causality(self):
        """ Future data in X should not affect past risk contributions. """
        rng = np.random.default_rng(48)
        T, N = 200, 5
        n = 50
        X = rng.standard_normal((T, N))
        W = rng.uniform(0.1, 1.0, (T, N))
        W /= W.sum(axis=1, keepdims=True)

        # Compute contributions
        rc_orig = roll_risk_contribution(W, X.copy(), n=n, pct=True)

        # Modify data after t0
        t0 = n + 7
        X_modified = X.copy()
        X_modified[t0:] *= 3.0

        rc_modified = roll_risk_contribution(W, X_modified, n=n, pct=True)

        # Rows strictly before t0 should be unchanged
        assert np.allclose(rc_orig[:t0], rc_modified[:t0], equal_nan=True)

    def test_roll_contributions_sum_to_one_pct(self):
        """ roll_risk_contribution with pct=True should have rows summing to 1. """
        rng = np.random.default_rng(49)
        T, N = 100, 5
        n = 20
        X = rng.standard_normal((T, N))
        W = rng.uniform(0.1, 1.0, (T, N))
        W /= W.sum(axis=1, keepdims=True)

        rc = roll_risk_contribution(W, X, n=n, pct=True)

        # Rows >= n should sum to ~1.0
        for t in range(n, T):
            assert np.allclose(rc[t].sum(), 1.0, rtol=1e-10)

    def test_roll_with_ledoit_wolf(self):
        """ roll_risk_contribution should work with ledoit_wolf covariance. """
        rng = np.random.default_rng(50)
        T, N = 100, 5
        n = 20
        X = rng.standard_normal((T, N))
        W = rng.uniform(0.1, 1.0, (T, N))
        W /= W.sum(axis=1, keepdims=True)

        rc = roll_risk_contribution(W, X, n=n, cov=ledoit_wolf, pct=True)

        assert rc.shape == (T, N)
        assert np.isnan(rc[:n]).all()
        # Rows >= n should sum to ~1.0
        for t in range(n, min(n + 10, T)):
            assert np.allclose(rc[t].sum(), 1.0, rtol=1e-3)


# =========================================================================== #
#                             Edge cases                                      #
# =========================================================================== #


class TestEdgeCases:
    """ Tests for edge cases and error handling. """

    def test_risk_contribution_validates_inputs(self):
        """ risk_contribution should validate input shapes and values. """
        w = np.array([0.5, 0.5])
        sigma = np.eye(2)

        # Non-finite w
        with pytest.raises(ValueError, match="non-finite"):
            risk_contribution(np.array([0.5, np.nan]), sigma)

        # Non-finite sigma
        with pytest.raises(ValueError, match="non-finite"):
            risk_contribution(w, np.array([[1.0, np.inf], [np.inf, 1.0]]))

    def test_roll_risk_contribution_validates_inputs(self):
        """ roll_risk_contribution should validate input shapes and window size. """
        X = np.ones((100, 3))
        W = np.ones((100, 3)) / 3.0

        # Shape mismatch
        with pytest.raises(ValueError, match="does not match"):
            roll_risk_contribution(W, X[:-1], n=20)

        # Invalid n
        with pytest.raises(ValueError, match="must be"):
            roll_risk_contribution(W, X, n=0)

        with pytest.raises(ValueError, match="must be"):
            roll_risk_contribution(W, X, n=100)
