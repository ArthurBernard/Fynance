#!/usr/bin/env python3
# coding: utf-8

""" Tests for rebalancing policies, lot discretization and delay.

Covers the drift law, the three rebalancing policies (calendar / band /
turnover cap), lot discretization and execution delay from
:mod:`fynance.portfolio.rebalance`: hand-checked numerics, per-function
causality probes and validation errors.
"""

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.portfolio.rebalance import (
    _drift_step,
    _returns,
    delay,
    discretize,
    rebalance_band,
    rebalance_calendar,
    rebalance_turnover_cap,
)

# =========================================================================== #
#                                drift law                                    #
# =========================================================================== #


class TestDrift:
    """ Hand-checked mark-to-market drift, 3 bars x 2 assets, exact fractions. """

    def test_drift_exact_fractions(self):
        # A large ``every`` disables calendar rebalancing after bar 0, so
        # calendar reduces to pure drift from the initial target W[0].
        W = np.full((3, 2), 0.5)
        X = np.array([
            [100.0, 100.0],
            [110.0, 80.0],   # r1 = [+0.1, -0.2]
            [121.0, 96.0],   # r2 = [+0.1, +0.2]
        ])
        E = rebalance_calendar(W, X, every=100)

        # Bar 1: book_ret = 0.5*0.1 + 0.5*(-0.2) = -0.05, denom = 0.95.
        #   w0 = 0.5*1.1/0.95 = 11/19 ; w1 = 0.5*0.8/0.95 = 8/19.
        # Bar 2: from [11/19, 8/19], r2 = [0.1, 0.2].
        #   book_ret = (11*0.1 + 8*0.2)/19 = 2.7/19, denom = 21.7/19.
        #   w0 = 11*1.1/21.7 = 121/217 ; w1 = 8*1.2/21.7 = 96/217.
        expected = np.array([
            [1 / 2, 1 / 2],
            [11 / 19, 8 / 19],
            [121 / 217, 96 / 217],
        ])
        assert np.allclose(E, expected, rtol=0.0, atol=1e-15)
        # Fully-invested long-only book: drift preserves sum-to-one.
        assert np.allclose(E.sum(axis=1), 1.0, atol=1e-15)

    def test_drift_wipeout_guard(self):
        # Book return <= -1 zeroes the weights from that bar on.
        w = np.array([1.0, 1.0])
        r = np.array([-1.0, -1.0])  # book_ret = -2 <= -1
        out = np.asarray(_drift_step(w, r))
        assert np.array_equal(out, np.zeros(2))

    def test_single_asset_fully_invested_is_static(self):
        # 100% in one asset stays 100% under any return (no drift).
        w = np.array([1.0])
        for ret in (-0.3, 0.0, 0.25, 1.0):
            out = np.asarray(_drift_step(w, np.array([ret])))
            assert np.isclose(out[0], 1.0)


# =========================================================================== #
#                             calendar schedule                              #
# =========================================================================== #


class TestCalendar:
    """ Weights jump to target only on bars that are multiples of ``every``. """

    def test_trades_only_on_schedule(self):
        rng = np.random.default_rng(0)
        T, N, every = 40, 3, 7
        W = rng.uniform(0.0, 1.0, size=(T, N))
        W /= W.sum(axis=1, keepdims=True)
        X = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(T, N)), axis=0)

        E = rebalance_calendar(W, X, every=every)
        R = _returns(X)

        assert np.allclose(E[0], W[0])
        for t in range(1, T):
            if t % every == 0:
                # Rebalance bar: snaps exactly to the target.
                assert np.allclose(E[t], W[t]), t

            else:
                # Non-rebalance bar: pure drift of the previous held book.
                expected = np.asarray(_drift_step(E[t - 1], R[t]))
                assert np.allclose(E[t], expected), t

    def test_every_one_is_raw_target(self):
        rng = np.random.default_rng(1)
        W = rng.uniform(0.0, 1.0, size=(15, 4))
        X = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(15, 4)), axis=0)

        E = rebalance_calendar(W, X, every=1)

        assert np.allclose(E, W)

    def test_1d_promotion_and_squeeze(self):
        W = np.full(10, 1.0)
        X = 100.0 * np.cumprod(1.0 + np.full(10, 0.01))
        E = rebalance_calendar(W, X, every=3)
        assert E.ndim == 1
        assert E.shape == (10,)
        # 100% single asset never drifts and always snaps back to 1.
        assert np.allclose(E, 1.0)


# =========================================================================== #
#                               no-trade band                                #
# =========================================================================== #


class TestBand:
    """ Band: no trading inside the band, edge mode lands on the boundary. """

    def test_no_trade_inside_band(self):
        # Small returns keep the drift within a wide band -> band never
        # triggers, so the book is exactly the pure-drift book.
        rng = np.random.default_rng(2)
        T, N = 12, 2
        W = np.full((T, N), 0.5)
        X = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, size=(T, N)), axis=0)
        band = 0.3

        E_band = rebalance_band(W, X, band=band, mode='full')
        E_drift = rebalance_calendar(W, X, every=10 ** 9)  # never rebalances

        # Confirm the hand-built path really stays inside the band...
        assert np.max(np.abs(E_drift - W)) < band
        # ...so the band policy did zero trading (equals pure drift).
        assert np.allclose(E_band, E_drift)

    def test_full_snaps_back_to_target(self):
        W = np.array([[0.5, 0.5], [0.5, 0.5]])
        X = np.array([[100.0, 100.0], [120.0, 80.0]])  # r = [+0.2, -0.2]

        E = rebalance_band(W, X, band=0.05, mode='full')

        # drift -> [0.6, 0.4], dev 0.1 > 0.05 -> full trade back to target.
        assert np.allclose(E[1], [0.5, 0.5])

    def test_edge_lands_on_boundary(self):
        W = np.array([[0.5, 0.5], [0.5, 0.5]])
        X = np.array([[100.0, 100.0], [120.0, 80.0]])

        E = rebalance_band(W, X, band=0.05, mode='edge')

        # drift -> [0.6, 0.4] clipped to [0.45, 0.55] -> exactly on the edges.
        assert np.allclose(E[1], [0.55, 0.45])
        assert np.isclose(E[1, 0], W[1, 0] + 0.05)
        assert np.isclose(E[1, 1], W[1, 1] - 0.05)


# =========================================================================== #
#                              turnover budget                               #
# =========================================================================== #


class TestTurnoverCap:
    """ Per-bar turnover never exceeds the budget; the book converges. """

    def test_per_bar_turnover_and_convergence(self):
        T, N = 50, 2
        target = np.array([0.4, 0.6])
        W = np.tile(target, (T, 1))
        X = np.full((T, N), 100.0)  # constant prices -> no drift
        budget = 0.10

        E = rebalance_turnover_cap(W, X, budget=budget)

        # Turnover from a flat start: |E[0]| then |E[t] - E[t-1]| (no drift).
        book = np.vstack([np.zeros(N), E])
        turnover = np.abs(np.diff(book, axis=0)).sum(axis=1)
        assert np.all(turnover <= budget + 1e-12)

        # Gap shrinks by exactly ``budget`` per bar until it snaps to target.
        assert np.allclose(E[-1], target, atol=1e-12)

    def test_first_bar_capped_from_flat(self):
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        X = np.full((2, 2), 100.0)

        E = rebalance_turnover_cap(W, X, budget=0.5)

        # Desired entry turnover 1.0 > 0.5 -> scaled by 0.5.
        assert np.allclose(E[0], [0.5, 0.0])
        assert np.isclose(np.abs(E[0]).sum(), 0.5)

    def test_small_desired_move_executes_in_full(self):
        # When the desired move is under budget, trade all the way.
        target = np.array([0.3, 0.7])
        W = np.tile(target, (5, 1))
        X = np.full((5, 2), 100.0)

        E = rebalance_turnover_cap(W, X, budget=2.0)  # budget > full entry

        assert np.allclose(E, W)

    def test_cap_respected_with_drift(self):
        # With genuine drift, the traded amount (relative to the drifted book)
        # must still respect the budget at every bar.
        rng = np.random.default_rng(5)
        T, N = 60, 4
        W = rng.uniform(0.0, 1.0, size=(T, N))
        W /= W.sum(axis=1, keepdims=True)
        X = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=(T, N)), axis=0)
        budget = 0.15

        E = rebalance_turnover_cap(W, X, budget=budget)
        R = _returns(X)

        prev = np.zeros(N)
        for t in range(T):
            drift = np.zeros(N) if t == 0 else np.asarray(_drift_step(prev, R[t]))
            traded = np.abs(E[t] - drift).sum()
            assert traded <= budget + 1e-12, t
            prev = E[t]


# =========================================================================== #
#                            lot discretization                              #
# =========================================================================== #


class TestDiscretize:
    """ Exact share/lot math, min_notional suppression, round lots. """

    def test_exact_share_lot_math(self):
        # capital 1000, unit lots: 0.5 target on a $300 asset -> round(500/300)
        # = 2 shares -> 0.6 weight; the $50 asset lands exactly on 0.5.
        W = np.array([[0.5, 0.5]])
        prices = np.array([[300.0, 50.0]])

        E = discretize(W, prices, capital=1000.0, lot=1.0)

        assert np.allclose(E, [[0.6, 0.5]])

    def test_round_lots_of_100(self):
        # capital 1e6, price 150, lot 100: round(0.5e6/150 / 100)*100 = 3300
        # shares -> 3300*150/1e6 = 0.495.
        W = np.array([[0.5]])
        prices = np.array([[150.0]])

        E = discretize(W, prices, capital=1e6, lot=100.0)

        assert np.allclose(E, [[0.495]])

    def test_min_notional_suppresses_small_trade(self):
        # Bar 0 buys 5 shares (0.5). Bar 1 target 0.56 -> 6 shares, a 1-share
        # ($100) rebalance trade.
        W = np.array([[0.5], [0.56]])
        prices = np.array([[100.0], [100.0]])

        # min_notional above the $100 trade -> suppressed, hold 5 shares.
        E_sup = discretize(W, prices, capital=1000.0, lot=1.0, min_notional=150.0)
        assert np.allclose(E_sup[:, 0], [0.5, 0.5])

        # min_notional below the trade -> it executes, 6 shares.
        E_exec = discretize(W, prices, capital=1000.0, lot=1.0, min_notional=50.0)
        assert np.allclose(E_exec[:, 0], [0.5, 0.6])

    def test_long_short_negative_lots(self):
        # A short target rounds to a negative share count.
        W = np.array([[-0.5, 0.5]])
        prices = np.array([[100.0, 100.0]])

        E = discretize(W, prices, capital=1000.0, lot=1.0)

        assert np.allclose(E, [[-0.5, 0.5]])


# =========================================================================== #
#                              execution delay                               #
# =========================================================================== #


class TestDelay:
    """ Shift the book by ``steps`` bars with a zero head. """

    def test_shift_and_zero_head(self):
        W = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])

        E = delay(W, steps=1)

        expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        assert np.array_equal(E, expected)

    def test_multi_step(self):
        W = np.arange(10.0).reshape(5, 2)

        E = delay(W, steps=2)

        assert np.array_equal(E[:2], np.zeros((2, 2)))
        assert np.array_equal(E[2:], W[:3])

    def test_zero_steps_is_copy(self):
        W = np.arange(6.0).reshape(3, 2)
        E = delay(W, steps=0)
        assert np.array_equal(E, W)
        assert E is not W  # a fresh array, not the input

    def test_steps_ge_T_all_zero(self):
        W = np.ones((4, 3))
        E = delay(W, steps=4)
        assert np.array_equal(E, np.zeros((4, 3)))
        E2 = delay(W, steps=10)
        assert np.array_equal(E2, np.zeros((4, 3)))

    def test_1d_squeeze(self):
        W = np.array([1.0, 2.0, 3.0])
        E = delay(W, steps=1)
        assert E.ndim == 1
        assert np.array_equal(E, [0.0, 1.0, 2.0])


# =========================================================================== #
#                              causality probes                              #
# =========================================================================== #


class TestCausality:
    """ Perturbing X[t0:] and W[t0:] must not touch any output before t0. """

    def _panel(self, seed, T=60, N=4):
        rng = np.random.default_rng(seed)
        W = rng.uniform(0.0, 1.0, size=(T, N))
        W /= W.sum(axis=1, keepdims=True)
        X = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(T, N)), axis=0)

        return W, X

    def _perturb(self, W, X, t0):
        W2 = W.copy()
        X2 = X.copy()
        W2[t0:] = 0.25          # arbitrary alternate targets
        X2[t0:] *= 1.5          # arbitrary alternate price path

        return W2, X2

    @pytest.mark.parametrize("fn,kw", [
        (rebalance_calendar, {'every': 7}),
        (rebalance_band, {'band': 0.02, 'mode': 'full'}),
        (rebalance_band, {'band': 0.02, 'mode': 'edge'}),
        (rebalance_turnover_cap, {'budget': 0.1}),
    ])
    def test_wx_policies_causal(self, fn, kw):
        W, X = self._panel(seed=10)
        t0 = 30
        base = fn(W, X, **kw)
        W2, X2 = self._perturb(W, X, t0)
        pert = fn(W2, X2, **kw)

        assert np.array_equal(base[:t0], pert[:t0])
        # Sanity: the perturbation actually changes the tail.
        assert not np.array_equal(base[t0:], pert[t0:])

    def test_discretize_causal(self):
        W, X = self._panel(seed=11)
        t0 = 30
        base = discretize(W, X, capital=1e5, lot=1.0, min_notional=10.0)
        W2, X2 = self._perturb(W, X, t0)
        pert = discretize(W2, X2, capital=1e5, lot=1.0, min_notional=10.0)

        assert np.array_equal(base[:t0], pert[:t0])
        assert not np.array_equal(base[t0:], pert[t0:])

    def test_delay_causal(self):
        W, _ = self._panel(seed=12)
        t0 = 30
        base = delay(W, steps=3)
        W2 = W.copy()
        W2[t0:] = 0.25
        pert = delay(W2, steps=3)

        assert np.array_equal(base[:t0], pert[:t0])


# =========================================================================== #
#                          shape / validation errors                         #
# =========================================================================== #


class TestValidation:
    """ Shape mismatches and out-of-range parameters raise ValueError. """

    def test_shape_mismatch(self):
        W = np.ones((10, 3))
        X = np.ones((10, 2))
        with pytest.raises(ValueError, match="same shape"):
            rebalance_calendar(W, X)

    def test_non_finite(self):
        W = np.ones((5, 2))
        X = np.ones((5, 2))
        X[2, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            rebalance_calendar(W, X)

    def test_ndim_error(self):
        W = np.ones((3, 2, 2))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            delay(W)

    def test_calendar_every_too_small(self):
        W = np.ones((5, 2))
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="every must be >= 1"):
            rebalance_calendar(W, X, every=0)

    def test_band_bad_mode(self):
        W = np.ones((5, 2))
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="mode"):
            rebalance_band(W, X, mode='partial')

    def test_band_negative(self):
        W = np.ones((5, 2))
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="band must be >= 0"):
            rebalance_band(W, X, band=-0.1)

    def test_cap_negative_budget(self):
        W = np.ones((5, 2))
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="budget must be >= 0"):
            rebalance_turnover_cap(W, X, budget=-0.1)

    @pytest.mark.parametrize("kw,msg", [
        ({'capital': 0.0}, "capital must be > 0"),
        ({'lot': 0.0}, "lot must be > 0"),
        ({'min_notional': -1.0}, "min_notional must be >= 0"),
    ])
    def test_discretize_bad_params(self, kw, msg):
        W = np.ones((5, 2))
        prices = np.full((5, 2), 100.0)
        with pytest.raises(ValueError, match=msg):
            discretize(W, prices, **kw)

    def test_delay_negative_steps(self):
        W = np.ones((5, 2))
        with pytest.raises(ValueError, match="steps must be >= 0"):
            delay(W, steps=-1)
