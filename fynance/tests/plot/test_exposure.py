#!/usr/bin/env python3
# coding: utf-8

""" Headless tests for the gross/net exposure figure. """

# Built-in packages
from types import SimpleNamespace

# Third-party packages
import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# Local packages
from fynance.plot import plot_exposure, tearsheet  # noqa: E402


def test_plot_exposure_returns_axes_with_gross_and_net_lines():
    W = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, -1.0], [0.0, 0.0]])
    ax = plot_exposure(W)
    assert ax is not None
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "gross" in labels
    assert "net" in labels
    gross_line = ax.get_lines()[labels.index("gross")]
    net_line = ax.get_lines()[labels.index("net")]
    assert np.allclose(gross_line.get_ydata(), [1.0, 1.0, 2.0, 0.0])
    assert np.allclose(net_line.get_ydata(), [1.0, 0.0, -2.0, 0.0])
    plt.close("all")


def test_plot_exposure_1d_promotion():
    w = np.array([1.0, -1.0, 0.0])
    ax = plot_exposure(w)
    labels = [ln.get_label() for ln in ax.get_lines()]
    gross_line = ax.get_lines()[labels.index("gross")]
    net_line = ax.get_lines()[labels.index("net")]
    assert np.allclose(gross_line.get_ydata(), [1.0, 1.0, 0.0])
    assert np.allclose(net_line.get_ydata(), [1.0, -1.0, 0.0])
    plt.close("all")


def test_plot_exposure_save_writes_file(tmp_path):
    W = np.array([[1.0, 0.0], [0.5, -0.5], [-1.0, -1.0], [0.0, 0.0]])
    out = tmp_path / "exposure.png"
    ax = plot_exposure(W)
    ax.figure.savefig(out)
    assert out.is_file()
    assert out.stat().st_size > 0
    plt.close("all")


def _positions_result(n=60, n_assets=3, seed=0):
    """ A duck-typed BacktestResult-like object carrying a positions book. """
    rng = np.random.default_rng(seed)
    positions = rng.choice([-1.0, 0.0, 1.0], size=(n, n_assets))
    equity = np.cumprod(1.0 + rng.normal(0.0, 0.01, n))

    return SimpleNamespace(equity=equity, positions=positions, index=None)


def test_tearsheet_show_exposure_default_false_stays_unchanged():
    res = _positions_result()
    fig_default = tearsheet(res)
    fig_off = tearsheet(res, show_exposure=False)
    assert len(fig_default.axes) == len(fig_off.axes)
    plt.close("all")


def test_tearsheet_show_exposure_adds_one_panel():
    res = _positions_result()
    fig_off = tearsheet(res, show_exposure=False)
    fig_on = tearsheet(res, show_exposure=True)
    assert len(fig_on.axes) == len(fig_off.axes) + 1
    titles = " ".join(ax.get_title().lower() for ax in fig_on.axes)
    assert "exposure" in titles
    plt.close("all")


def test_tearsheet_show_exposure_noop_without_positions():
    equity = np.cumprod(1.0 + np.random.default_rng(0).normal(0.0, 0.01, 100))
    fig = tearsheet(equity, show_exposure=True)
    assert len(fig.axes) == 4
    plt.close("all")


def test_import_fynance_plot_exposure_stays_matplotlib_free():
    import subprocess
    import sys

    code = ("import fynance.plot.exposure, sys; "
            "print('matplotlib' in sys.modules)")
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "False"
