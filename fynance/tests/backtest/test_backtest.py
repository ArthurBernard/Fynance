#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pytest

from fynance.backtest.loss import LossSeries
from fynance.backtest.print_stats import set_text_stats


# =========================================================================== #
#                              LossSeries tests                                #
# =========================================================================== #


def test_loss_series_append_float():
    ls = LossSeries()
    ls.append(0.5)
    assert ls.values.shape == (1,)
    assert ls.values[0] == pytest.approx(0.5)


def test_loss_series_append_int():
    ls = LossSeries()
    ls.append(3)
    assert ls.values.shape == (1,)
    assert ls.values[0] == pytest.approx(3.0)


def test_loss_series_append_list():
    ls = LossSeries()
    ls.append([1.0, 2.0, 3.0])
    assert ls.values.shape == (3,)


def test_loss_series_append_multiple():
    ls = LossSeries()
    ls.append(1.0)
    ls.append(2.0)
    assert ls.values.shape == (2,)


def test_loss_series_reset():
    ls = LossSeries()
    ls.append([1.0, 2.0, 3.0])
    ls.reset()
    assert ls.values.shape == (0,)


def test_loss_series_repr():
    ls = LossSeries()
    ls.append(1.0)
    assert repr(ls).startswith('LossSeries(')


def test_loss_series_str():
    ls = LossSeries()
    ls.append(1.0)
    assert isinstance(str(ls), str)


# =========================================================================== #
#                            set_text_stats tests                              #
# =========================================================================== #


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.01, 252)


def test_set_text_stats_returns_string(synthetic_returns):
    pred = np.where(synthetic_returns > 0, 1.0, -1.0)
    txt = set_text_stats(synthetic_returns, strategy=pred)
    assert isinstance(txt, str)
    assert len(txt) > 0


def test_set_text_stats_contains_sections(synthetic_returns):
    pred = np.where(synthetic_returns > 0, 1.0, -1.0)
    txt = set_text_stats(synthetic_returns, strategy=pred)
    assert 'Accuracy' in txt
    assert 'Performance' in txt
    assert 'Sharpe' in txt
    assert 'Calmar' in txt


def test_set_text_stats_accur_false(synthetic_returns):
    pred = np.where(synthetic_returns > 0, 1.0, -1.0)
    txt = set_text_stats(synthetic_returns, accur=False, strategy=pred)
    assert 'Accuracy' not in txt


def test_set_text_stats_vol_false(synthetic_returns):
    pred = np.where(synthetic_returns > 0, 1.0, -1.0)
    txt = set_text_stats(synthetic_returns, vol=False, strategy=pred)
    assert 'Volatility' not in txt


def test_set_text_stats_no_strategies(synthetic_returns):
    txt = set_text_stats(synthetic_returns)
    assert isinstance(txt, str)
    assert 'Performance' in txt
