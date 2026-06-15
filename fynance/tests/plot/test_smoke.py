#!/usr/bin/env python3
# coding: utf-8

""" Headless smoke tests for the plotting layer. """

# Third-party packages
import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# Local packages
from fynance.backtest import backtest  # noqa: E402
from fynance.plot import (  # noqa: E402
    plot_drawdown,
    plot_equity,
    plot_returns_hist,
    plot_rolling_sharpe,
    tearsheet,
    tearsheet_text,
)


def _result():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 300)
    positions = np.sign(rng.normal(size=300))
    return backtest(returns, positions, shift=True)


def test_plot_functions_return_axes():
    res = _result()
    for fn in (plot_equity, plot_drawdown, plot_returns_hist):
        ax = fn(res)
        assert ax is not None
        plt.close("all")
    ax = plot_rolling_sharpe(res, window=50)
    assert ax is not None
    plt.close("all")


def test_tearsheet_returns_figure_with_axes():
    res = _result()
    fig = tearsheet(res, period=50)
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_tearsheet_text_matches_summary():
    res = _result()
    txt = tearsheet_text(res)
    assert "sharpe" in txt
    assert "max_drawdown" in txt
