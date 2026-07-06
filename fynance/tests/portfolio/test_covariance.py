""" Tests for conditioned covariance estimators (fynance.portfolio.covariance). """

import numpy as np
import pytest

from fynance.portfolio.covariance import (
    _lw_intensity,
    denoise_cov,
    ewma_cov,
    factor_cov,
    ledoit_wolf,
    sample_cov,
)

# ---------------------------------------------------------------------------
# Slow, obviously-correct reference implementations
# ---------------------------------------------------------------------------


def _ref_ewma_cov(X: np.ndarray, halflife: float) -> np.ndarray:
    """ Explicit-loop RiskMetrics-style EWMA covariance (reference). """
    T, N = X.shape
    lam = 0.5 ** (1.0 / halflife)
    mean = X.mean(axis=0)
    weights = [lam ** (T - 1 - t) for t in range(T)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    cov = np.zeros((N, N))
    for t in range(T):
        x = X[t] - mean
        for i in range(N):
            for j in range(N):
                cov[i, j] += weights[t] * x[i] * x[j]

    return cov


def _ref_lw_const_corr_intensity(X: np.ndarray) -> float:
    """ Explicit-loop Ledoit & Wolf (2004b) constant-correlation intensity. """
    T, N = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    S = np.zeros((N, N))
    for t in range(T):
        for i in range(N):
            for j in range(N):
                S[i, j] += Xc[t, i] * Xc[t, j]
    S /= T

    d = np.sqrt(np.diag(S))
    r_bar_num = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                r_bar_num += S[i, j] / (d[i] * d[j])
    r_bar = r_bar_num / (N * (N - 1))

    F = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            F[i, j] = S[i, i] if i == j else r_bar * d[i] * d[j]

    pi_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            acc = 0.0
            for t in range(T):
                acc += (Xc[t, i] * Xc[t, j] - S[i, j]) ** 2
            pi_mat[i, j] = acc / T
    pi_hat = pi_mat.sum()

    rho_hat = np.trace(pi_mat)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            theta_ii = 0.0
            theta_jj = 0.0
            for t in range(T):
                theta_ii += (Xc[t, i] ** 2 - S[i, i]) * (Xc[t, i] * Xc[t, j] - S[i, j])
                theta_jj += (Xc[t, j] ** 2 - S[j, j]) * (Xc[t, i] * Xc[t, j] - S[i, j])
            theta_ii /= T
            theta_jj /= T
            rho_hat += 0.5 * r_bar * (
                np.sqrt(S[j, j] / S[i, i]) * theta_ii
                + np.sqrt(S[i, i] / S[j, j]) * theta_jj
            )

    gamma_hat = np.sum((F - S) ** 2)
    kappa_hat = (pi_hat - rho_hat) / gamma_hat

    return float(np.clip(kappa_hat / T, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Reference-implementation parity
# ---------------------------------------------------------------------------


def test_ewma_cov_matches_reference_loop():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 5)) * 0.01
    got = ewma_cov(X, halflife=30.0)
    ref = _ref_ewma_cov(X, halflife=30.0)
    assert np.allclose(got, ref, rtol=1e-10)


def test_lw_const_corr_intensity_matches_reference_loop():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 5)) * 0.01
    got = _lw_intensity(X, target='const_corr')
    ref = _ref_lw_const_corr_intensity(X)
    assert np.isclose(got, ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# Properties on seeded (T, N) grids
# ---------------------------------------------------------------------------

GRIDS = [(30, 2), (30, 8), (300, 2), (300, 8)]


@pytest.mark.parametrize("T,N", GRIDS)
def test_sample_cov_properties(T, N):
    rng = np.random.default_rng(100 + T + N)
    X = rng.standard_normal((T, N)) * 0.01
    S = sample_cov(X)
    assert S.shape == (N, N)
    assert np.allclose(S, S.T)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-10)


@pytest.mark.parametrize("T,N", GRIDS)
@pytest.mark.parametrize("target", ["identity", "const_corr", "diag"])
def test_ledoit_wolf_properties(T, N, target):
    rng = np.random.default_rng(200 + T + N)
    X = rng.standard_normal((T, N)) * 0.01
    S = ledoit_wolf(X, target=target)
    assert S.shape == (N, N)
    assert np.allclose(S, S.T)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-10)

    intensity = _lw_intensity(X, target=target)
    assert -1e-12 <= intensity <= 1.0 + 1e-12


@pytest.mark.parametrize("T,N", GRIDS)
def test_ewma_cov_properties(T, N):
    rng = np.random.default_rng(300 + T + N)
    X = rng.standard_normal((T, N)) * 0.01
    S = ewma_cov(X, halflife=63.0)
    assert S.shape == (N, N)
    assert np.allclose(S, S.T)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-10)


@pytest.mark.parametrize("T,N", GRIDS)
def test_factor_cov_properties(T, N):
    rng = np.random.default_rng(400 + T + N)
    X = rng.standard_normal((T, N)) * 0.01
    S = factor_cov(X, n_factors=3)
    assert S.shape == (N, N)
    assert np.allclose(S, S.T)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-10)

    ref_diag = np.diag(sample_cov(X, ddof=0))
    assert np.allclose(np.diag(S), ref_diag, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("T,N", GRIDS)
def test_denoise_cov_properties(T, N):
    rng = np.random.default_rng(500 + T + N)
    X = rng.standard_normal((T, N)) * 0.01
    S = sample_cov(X, ddof=0)
    D = denoise_cov(S, n_obs=T)
    assert D.shape == (N, N)
    assert np.allclose(D, D.T)
    eigvals = np.linalg.eigvalsh(D)
    assert np.all(eigvals >= -1e-10)

    assert np.allclose(np.diag(D), np.diag(S), atol=1e-8)
    assert np.isclose(np.trace(D), np.trace(S), atol=1e-8)


# ---------------------------------------------------------------------------
# Limiting cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["identity", "const_corr", "diag"])
def test_ledoit_wolf_shrinks_less_as_t_grows(target):
    # Non-spherical, non-constant-correlation population: heterogeneous vols
    # and a mixed correlation structure, so the true covariance never
    # coincides exactly with any of the shrinkage targets (which would make
    # full shrinkage asymptotically optimal and break monotonicity).
    rng = np.random.default_rng(7)
    N = 5
    vols = np.array([0.01, 0.02, 0.015, 0.03, 0.008])
    corr = np.full((N, N), 0.3)
    np.fill_diagonal(corr, 1.0)
    corr[0, 1] = corr[1, 0] = 0.7
    sigma = corr * np.outer(vols, vols)

    X_small = rng.multivariate_normal(np.zeros(N), sigma, size=30)
    X_large = rng.multivariate_normal(np.zeros(N), sigma, size=5000)

    delta_small = _lw_intensity(X_small, target=target)
    delta_large = _lw_intensity(X_large, target=target)
    assert delta_large < delta_small


def test_ewma_cov_huge_halflife_matches_sample_cov_ddof0():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((300, 4)) * 0.01
    got = ewma_cov(X, halflife=1e6)
    ref = sample_cov(X, ddof=0)
    # lam = 0.5 ** (1 / 1e6) is extremely close to but not exactly 1, so
    # weights are near-uniform rather than bit-identical to 1/T; tolerance
    # is scaled to the data (~1e-4 covariance entries) with headroom for
    # that residual near-uniformity.
    scale = np.abs(ref).max()
    assert np.allclose(got, ref, atol=1e-5 * scale, rtol=1e-3)


def test_factor_cov_full_rank_matches_sample_cov_ddof0():
    rng = np.random.default_rng(9)
    N = 6
    X = rng.standard_normal((300, N)) * 0.01
    got = factor_cov(X, n_factors=N)
    ref = sample_cov(X, ddof=0)
    assert np.allclose(got, ref, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("target", ["identity", "const_corr", "diag"])
def test_n1_returns_variance_for_every_estimator(target):
    rng = np.random.default_rng(10)
    X = rng.standard_normal((100, 1)) * 0.01
    ref_var = sample_cov(X, ddof=1)

    assert np.allclose(sample_cov(X, ddof=1), ref_var)
    assert np.allclose(ledoit_wolf(X, target=target), sample_cov(X, ddof=0), rtol=1e-9)
    assert np.allclose(ewma_cov(X, halflife=1e6), sample_cov(X, ddof=0), rtol=1e-6)
    assert np.allclose(factor_cov(X, n_factors=1), sample_cov(X, ddof=0), rtol=1e-9)
    S0 = sample_cov(X, ddof=0)
    assert np.allclose(denoise_cov(S0, n_obs=100), S0, rtol=1e-9)


# ---------------------------------------------------------------------------
# Scale equivariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [2.0, 0.1, 5.0])
def test_scale_equivariance(c):
    rng = np.random.default_rng(11)
    X = rng.standard_normal((150, 5)) * 0.01

    assert np.allclose(sample_cov(c * X), c ** 2 * sample_cov(X), rtol=1e-9)
    assert np.allclose(
        ewma_cov(c * X, halflife=40.0), c ** 2 * ewma_cov(X, halflife=40.0), rtol=1e-9
    )
    assert np.allclose(
        factor_cov(c * X, n_factors=3), c ** 2 * factor_cov(X, n_factors=3), rtol=1e-9
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_nan_input_raises():
    X = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        sample_cov(X)
    with pytest.raises(ValueError):
        ledoit_wolf(X)
    with pytest.raises(ValueError):
        ewma_cov(X)
    with pytest.raises(ValueError):
        factor_cov(X)


def test_inf_input_raises():
    X = np.array([[1.0, np.inf], [2.0, 1.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        sample_cov(X)


def test_denoise_cov_bogus_method_raises():
    S = np.eye(3)
    with pytest.raises(ValueError):
        denoise_cov(S, n_obs=100, method='bogus')


def test_ledoit_wolf_bogus_target_raises():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((50, 3))
    with pytest.raises(ValueError):
        ledoit_wolf(X, target='bogus')


def test_factor_cov_bad_n_factors_raises():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((50, 3))
    with pytest.raises(ValueError):
        factor_cov(X, n_factors=0)


def test_1d_input_reshaped():
    rng = np.random.default_rng(14)
    x = rng.standard_normal(100) * 0.01
    S = sample_cov(x)
    assert S.shape == (1, 1)
