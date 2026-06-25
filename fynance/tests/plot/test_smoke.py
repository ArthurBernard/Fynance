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


def test_plot_equity_base_rescales_start():
    # base=100 must rescale the displayed curve to start at 100 (display only).
    res = _result()
    ax = plot_equity(res, base=100.0)
    ydata = ax.get_lines()[0].get_ydata()
    assert abs(float(ydata[0]) - 100.0) < 1e-9
    plt.close("all")


def test_plot_equity_logy_auto_switches_on_wide_amplitude():
    # A x20 ramp trips the auto log-scale; a near-flat curve stays linear.
    ax = plot_equity(np.linspace(1.0, 20.0, 100))
    assert ax.get_yscale() == "log"
    plt.close("all")

    ax = plot_equity(np.linspace(1.0, 1.2, 100))
    assert ax.get_yscale() == "linear"
    plt.close("all")


def test_plot_equity_logy_explicit_overrides_auto():
    # logy=False forces linear even when auto would have gone log.
    ax = plot_equity(np.linspace(1.0, 20.0, 100), logy=False)
    assert ax.get_yscale() == "linear"
    plt.close("all")
    # logy=True forces log even on a flat curve (still strictly positive).
    ax = plot_equity(np.linspace(1.0, 1.2, 100), logy=True)
    assert ax.get_yscale() == "log"
    plt.close("all")
