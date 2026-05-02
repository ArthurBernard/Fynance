#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from fynance.features.indicators import (
    bollinger_band,
    cci,
    hma,
    macd_hist,
    macd_line,
    rsi,
    signal_line,
)

N = 100
rng = np.random.default_rng(42)
X = np.cumsum(rng.normal(0, 1, N)) + 100.0


# =========================================================================== #
#                            bollinger_band                                    #
# =========================================================================== #


def test_bollinger_band_shape():
    upper, lower = bollinger_band(X, w=20)
    assert upper.shape == (N,)
    assert lower.shape == (N,)


def test_bollinger_band_upper_ge_lower():
    upper, lower = bollinger_band(X, w=20)
    assert np.all(upper >= lower)


def test_bollinger_band_no_nan_after_warmup():
    upper, lower = bollinger_band(X, w=10)
    assert not np.any(np.isnan(upper[10:]))
    assert not np.any(np.isnan(lower[10:]))


# =========================================================================== #
#                                  cci                                         #
# =========================================================================== #


def test_cci_shape():
    result = cci(X, w=20)
    assert result.shape == (N,)


def test_cci_no_nan_after_warmup():
    result = cci(X, w=10)
    assert not np.any(np.isnan(result[10:]))


# =========================================================================== #
#                                  hma                                         #
# =========================================================================== #


def test_hma_shape():
    result = hma(X, w=21)
    assert result.shape == (N,)


def test_hma_no_nan_after_warmup():
    result = hma(X, w=10)
    assert not np.any(np.isnan(result[10:]))


# =========================================================================== #
#                                macd_hist                                     #
# =========================================================================== #


def test_macd_hist_shape():
    result = macd_hist(X)
    assert result.shape == (N,)


def test_macd_hist_no_nan_after_warmup():
    result = macd_hist(X, w=9, fast_w=12, slow_w=26)
    assert not np.any(np.isnan(result[26:]))


# =========================================================================== #
#                                macd_line                                     #
# =========================================================================== #


def test_macd_line_shape():
    result = macd_line(X)
    assert result.shape == (N,)


def test_macd_line_no_nan_after_warmup():
    result = macd_line(X, fast_w=12, slow_w=26)
    assert not np.any(np.isnan(result[26:]))


# =========================================================================== #
#                                  rsi                                         #
# =========================================================================== #


def test_rsi_shape():
    result = rsi(X, w=14)
    assert result.shape == (N,)


def test_rsi_range():
    result = rsi(X, w=14)
    valid = result[~np.isnan(result)]
    assert np.all(valid >= 0.0)
    assert np.all(valid <= 100.0)


def test_rsi_no_nan_after_warmup():
    result = rsi(X, w=14)
    assert not np.any(np.isnan(result[14:]))


# =========================================================================== #
#                               signal_line                                    #
# =========================================================================== #


def test_signal_line_shape():
    result = signal_line(X)
    assert result.shape == (N,)


def test_signal_line_no_nan_after_warmup():
    result = signal_line(X, w=9, fast_w=12, slow_w=26)
    assert not np.any(np.isnan(result[26:]))
