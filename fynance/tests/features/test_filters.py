#!/usr/bin/env python3
# coding: utf-8

""" Tests for fynance.features.filters. """

import numpy as np
import pytest

from fynance.features.filters import (
    fit_kalman,
    kalman_filter,
    kalman_loglikelihood,
    rts_smoother,
)

RNG = np.random.default_rng(42)
N = 1
T = 100
G = F = np.eye(N)
W_TRUE = np.eye(N) * 0.5
V_TRUE = np.eye(N) * 1.0

_x = np.cumsum(RNG.multivariate_normal(np.zeros(N), W_TRUE, T), axis=0)
Y = _x + RNG.multivariate_normal(np.zeros(N), V_TRUE, T)


class TestKalmanFilter:

    def test_output_shapes(self):
        m, C, a, R, e, S = kalman_filter(Y, G, F, W_TRUE, V_TRUE)
        assert m.shape == (T, N)
        assert C.shape == (T, N, N)
        assert a.shape == (T, N)
        assert R.shape == (T, N, N)
        assert e.shape == (T, N)
        assert S.shape == (T, N, N)

    def test_covariances_positive_definite(self):
        _, C, _, R, _, S = kalman_filter(Y, G, F, W_TRUE, V_TRUE)
        for t in range(T):
            assert np.all(np.linalg.eigvalsh(C[t]) > 0)
            assert np.all(np.linalg.eigvalsh(R[t]) > 0)
            assert np.all(np.linalg.eigvalsh(S[t]) > 0)

    def test_filtered_mean_closer_than_prior(self):
        # Filtered mean should on average be closer to observations than the prior
        m, _, a, _, _, _ = kalman_filter(Y, G, F, W_TRUE, V_TRUE)
        err_filtered = np.mean((m - Y) ** 2)
        err_prior = np.mean((a - Y) ** 2)
        assert err_filtered <= err_prior

    def test_custom_init(self):
        m0 = np.zeros(N)
        C0 = np.eye(N) * 10.0
        m, C, _, _, _, _ = kalman_filter(Y, G, F, W_TRUE, V_TRUE, m0=m0, C0=C0)
        assert m.shape == (T, N)

    def test_1d_input_raises(self):
        with pytest.raises((ValueError, IndexError)):
            kalman_filter(Y.ravel(), G, F, W_TRUE, V_TRUE)


class TestRTSSmoother:

    def _filter(self):
        return kalman_filter(Y, G, F, W_TRUE, V_TRUE)

    def test_output_shapes(self):
        m, C, a, R, e, S = self._filter()
        ms, Cs = rts_smoother(m, C, a, R, G)
        assert ms.shape == (T, N)
        assert Cs.shape == (T, N, N)

    def test_smoother_last_equals_filter(self):
        m, C, a, R, _, _ = self._filter()
        ms, Cs = rts_smoother(m, C, a, R, G)
        np.testing.assert_allclose(ms[-1], m[-1])
        np.testing.assert_allclose(Cs[-1], C[-1])

    def test_smoother_reduces_variance(self):
        # Smoothed covariances should be <= filtered covariances (elementwise)
        m, C, a, R, _, _ = self._filter()
        ms, Cs = rts_smoother(m, C, a, R, G)
        assert np.all(Cs[:-1] <= C[:-1] + 1e-10)


class TestKalmanLoglikelihood:

    def test_returns_float(self):
        _, _, _, _, e, S = kalman_filter(Y, G, F, W_TRUE, V_TRUE)
        ll = kalman_loglikelihood(e, S)
        assert isinstance(ll, float)

    def test_true_params_better_than_wrong(self):
        _, _, _, _, e_true, S_true = kalman_filter(Y, G, F, W_TRUE, V_TRUE)
        ll_true = kalman_loglikelihood(e_true, S_true)

        W_bad = np.eye(N) * 10.0
        V_bad = np.eye(N) * 10.0
        _, _, _, _, e_bad, S_bad = kalman_filter(Y, G, F, W_bad, V_bad)
        ll_bad = kalman_loglikelihood(e_bad, S_bad)

        assert ll_true > ll_bad


class TestFitKalman:

    def test_converges(self):
        result = fit_kalman(Y, G, F)
        assert result['success']

    def test_output_keys(self):
        result = fit_kalman(Y, G, F)
        assert set(result.keys()) == {'W', 'V', 'loglik', 'success'}

    def test_output_shapes(self):
        result = fit_kalman(Y, G, F)
        assert result['W'].shape == (N, N)
        assert result['V'].shape == (N, N)

    def test_estimated_loglik_better_than_identity(self):
        result = fit_kalman(Y, G, F)
        _, _, _, _, e_id, S_id = kalman_filter(Y, G, F, np.eye(N), np.eye(N))
        ll_id = kalman_loglikelihood(e_id, S_id)
        assert result['loglik'] >= ll_id - 1e-6

    def test_fit_only_V(self):
        result = fit_kalman(Y, G, F, fit_W=False, W0=W_TRUE)
        assert result['success']
        np.testing.assert_allclose(result['W'], W_TRUE)

    def test_fit_only_W(self):
        result = fit_kalman(Y, G, F, fit_V=False, V0=V_TRUE)
        assert result['success']
        np.testing.assert_allclose(result['V'], V_TRUE)

    def test_no_fit_returns_fixed(self):
        result = fit_kalman(Y, G, F, fit_W=False, fit_V=False, W0=W_TRUE, V0=V_TRUE)
        assert result['success']
        np.testing.assert_allclose(result['W'], W_TRUE)
        np.testing.assert_allclose(result['V'], V_TRUE)
