#!/usr/bin/env python3
# coding: utf-8

""" Tests for the trading-profile metrics (churn, turnover, exposure). """

# Third-party packages
import numpy as np

# Local packages
from fynance.metrics import (
    annual_turnover,
    exposure_summary,
    gross_exposure,
    net_exposure,
    sign_changes,
    trades_per_year,
    turnover_series,
)
from fynance.portfolio.sizing import transaction_cost


def test_sign_changes_counts_long_flat_short_transitions():
    # 1 -> 1 (no), 1 -> -1 (yes), -1 -> -1 (no), -1 -> 0 (yes), 0 -> 1 (yes).
    pos = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 1.0])
    assert sign_changes(pos) == 3


def test_sign_changes_constant_position_is_zero():
    assert sign_changes(np.ones(10)) == 0
    assert sign_changes(np.zeros(10)) == 0


def test_sign_changes_per_asset_book():
    book = np.array([[1.0, 0.0], [-1.0, 0.0], [-1.0, 1.0]])
    out = sign_changes(book)
    assert out.shape == (2,)
    assert list(out) == [1, 1]


def test_sign_changes_ignores_nan_straddling_pairs():
    # The pair (1.0, nan) and (nan, -1.0) straddle a NaN and are not counted;
    # only the eventual -1 -> 1 would count, which is absent here.
    pos = np.array([1.0, np.nan, -1.0])
    assert sign_changes(pos) == 0


def test_sign_changes_short_series():
    assert sign_changes(np.array([1.0])) == 0
    assert sign_changes(np.array([], dtype=float)) == 0


def test_trades_per_year_annualizes():
    pos = np.array([1.0, -1.0, 1.0, -1.0])  # 3 changes over 4 bars
    assert np.isclose(trades_per_year(pos, period=252), 3.0 / 4.0 * 252.0)


def test_trades_per_year_per_asset_book():
    book = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, 1.0]])  # asset0: 2, asset1: 0
    out = trades_per_year(book, period=252)
    assert out.shape == (2,)
    assert np.isclose(out[0], 2.0 / 3.0 * 252.0)
    assert out[1] == 0.0


# Hand-built 4-bar, 2-asset book exercising a long, a hedged-flat, a short and
# a flat-from-cover bar, so every branch of exposure_summary's pct_* is hit:
# bar0 net=+1 (long), bar1 net=0 (hedged), bar2 net=-2 (short), bar3 net=0 (flat).
_W = np.array([
    [1.0, 0.0],
    [0.5, -0.5],
    [-1.0, -1.0],
    [0.0, 0.0],
])


def test_turnover_series_hand_values():
    # bar0: |1|+|0|=1 ; bar1: |0.5-1|+|-0.5-0|=1 ; bar2: |-1-0.5|+|-1+0.5|=2 ;
    # bar3: |0+1|+|0+1|=2.
    assert np.allclose(turnover_series(_W), [1.0, 1.0, 2.0, 2.0])


def test_turnover_series_1d_promotion():
    w = np.array([0.0, 1.0, 1.0, -1.0])
    assert np.allclose(turnover_series(w), [0.0, 1.0, 0.0, 2.0])


def test_turnover_series_reconciles_with_transaction_cost():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(30, 5))
    fee = 0.0017
    lhs = turnover_series(W) * fee
    rhs = transaction_cost(W, fee=fee)
    assert np.allclose(lhs, rhs, rtol=1e-12)


def test_annual_turnover_hand_value():
    # mean([1, 1, 2, 2]) * 252 = 1.5 * 252 = 378.0
    assert np.isclose(annual_turnover(_W, period=252), 378.0)


def test_gross_exposure_hand_values():
    assert np.allclose(gross_exposure(_W), [1.0, 1.0, 2.0, 0.0])


def test_net_exposure_hand_values():
    assert np.allclose(net_exposure(_W), [1.0, 0.0, -2.0, 0.0])


def test_gross_net_exposure_1d_promotion():
    w = np.array([1.0, -1.0, 0.0])
    assert np.allclose(gross_exposure(w), [1.0, 1.0, 0.0])
    assert np.allclose(net_exposure(w), [1.0, -1.0, 0.0])


def test_exposure_summary_keys_and_hand_values():
    out = exposure_summary(_W, period=252)
    assert set(out) == {
        'annual_turnover', 'mean_gross', 'max_gross', 'mean_net',
        'min_net', 'max_net', 'pct_long', 'pct_short', 'pct_flat',
    }
    assert np.isclose(out['annual_turnover'], 378.0)
    assert np.isclose(out['mean_gross'], 1.0)
    assert np.isclose(out['max_gross'], 2.0)
    assert np.isclose(out['mean_net'], -0.25)
    assert np.isclose(out['min_net'], -2.0)
    assert np.isclose(out['max_net'], 1.0)
    # bar0 long (+1), bar2 short (-2), bar1 & bar3 flat (0) -> 25/25/50.
    assert np.isclose(out['pct_long'], 25.0)
    assert np.isclose(out['pct_short'], 25.0)
    assert np.isclose(out['pct_flat'], 50.0)
    assert np.isclose(out['pct_long'] + out['pct_short'] + out['pct_flat'], 100.0)
