#!/usr/bin/env python3
# coding: utf-8

""" Tests for the trading-profile metrics (sign-change churn). """

# Third-party packages
import numpy as np

# Local packages
from fynance.metrics import sign_changes, trades_per_year


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
