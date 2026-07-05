#!/usr/bin/env python3
# coding: utf-8

""" Headless tests for the factor tear-sheet figures. """

# Third-party packages
import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# Local packages
from fynance.metrics import quantile_returns, roll_information_coefficient  # noqa: E402
from fynance.plot import (  # noqa: E402
    factor_tearsheet,
    plot_ic_decay,
    plot_ic_series,
    plot_quantile_returns,
)


def _panel(T=250, N=20, seed=0):
    """ Seeded GBM price panel and a trailing-return factor aligned to it. """
    rng = np.random.default_rng(seed)
    prices = 100. * np.cumprod(1. + rng.normal(0., 0.01, (T, N)), axis=0)
    # A simple 21-bar trailing return as the factor (NaN head).
    factor = np.full((T, N), np.nan)
    factor[21:] = prices[21:] / prices[:-21] - 1.

    return factor, prices


def test_plot_quantile_returns_returns_axes():
    factor, prices = _panel()
    fwd = prices[1:] / prices[:-1] - 1.
    res = quantile_returns(factor[:-1], fwd, n_quantiles=5)
    ax = plot_quantile_returns(res)
    assert ax is not None
    # One line per quantile.
    assert len(ax.get_lines()) == 5
    plt.close("all")


def test_plot_ic_series_returns_axes_with_overlay():
    factor, prices = _panel()
    fwd = prices[1:] / prices[:-1] - 1.
    ic = roll_information_coefficient(factor[:-1], fwd, w=63)
    ax = plot_ic_series(ic, w_smooth=21)
    assert ax is not None
    # Raw IC line + smoothed overlay both present (plus the zero reference).
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "IC" in labels
    assert any(lab.startswith("IC (MA") for lab in labels)
    plt.close("all")


def test_plot_ic_decay_returns_axes():
    decay = np.array([0.05, 0.03, 0.01, 0.0])
    ax = plot_ic_decay(decay, horizons=(1, 5, 10, 21))
    assert ax is not None
    assert len(ax.patches) == 4  # one bar per horizon
    plt.close("all")


def test_factor_tearsheet_builds_2x2_grid():
    factor, prices = _panel()
    fig = factor_tearsheet(factor, prices)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_factor_tearsheet_save_writes_file(tmp_path):
    factor, prices = _panel()
    out = tmp_path / "factor.png"
    fig = factor_tearsheet(factor, prices, save=str(out))
    assert out.is_file()
    assert out.stat().st_size > 0
    plt.close(fig)


def test_factor_tearsheet_respects_custom_grid_params():
    factor, prices = _panel()
    fig = factor_tearsheet(factor, prices, n_quantiles=3,
                           horizons=(1, 5), w=40, figsize=(8, 5))
    assert len(fig.axes) == 4
    plt.close(fig)


def test_import_fynance_plot_factor_stays_matplotlib_free():
    import subprocess
    import sys

    code = ("import fynance.plot.factor, sys; "
            "print('matplotlib' in sys.modules)")
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "False"
