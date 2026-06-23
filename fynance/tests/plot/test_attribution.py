#!/usr/bin/env python3
# coding: utf-8

""" Tests for per-asset attribution figures and the book tearsheet. """

# Built-in packages
from types import SimpleNamespace

# Third-party packages
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Local packages
from fynance.plot import plot_contribution, plot_turnover, tearsheet


def _book_result(n=120, n_assets=3, seed=0):
    """ A duck-typed multi-asset BacktestResult-like object. """
    rng = np.random.default_rng(seed)
    asset_gross = rng.normal(0.0, 0.01, (n, n_assets))
    gross = asset_gross.sum(axis=1)
    equity = np.cumprod(1.0 + gross)
    positions = rng.choice([-1.0, 0.0, 1.0], size=(n, n_assets))

    return SimpleNamespace(
        equity=equity, returns=gross, gross_returns=gross,
        positions=positions, asset_gross_returns=asset_gross, index=None,
    )


def test_plot_contribution_is_cumulative_per_asset():
    asset = np.array([[0.01, -0.02], [0.03, 0.01]])
    ax = plot_contribution(asset)
    lines = ax.get_lines()
    assert len(lines) == 2
    # The last cumulative value of each line is that asset's total contribution.
    assert np.isclose(lines[0].get_ydata()[-1], asset[:, 0].sum())
    assert np.isclose(lines[1].get_ydata()[-1], asset[:, 1].sum())


def test_plot_turnover_charges_entry_from_flat():
    pos = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ax = plot_turnover(pos)
    lines = ax.get_lines()
    assert len(lines) == 2
    # Asset 0: |1-0|, |1-1|, |0-1| = 1, 0, 1 (first step is entry from flat).
    assert np.allclose(lines[0].get_ydata(), [1.0, 0.0, 1.0])


def test_tearsheet_book_adds_attribution_panels():
    fig = tearsheet(_book_result())
    # 2x2 core report + contribution + turnover = 6 axes.
    assert len(fig.axes) == 6
    titles = " ".join(ax.get_title().lower() for ax in fig.axes)
    assert "contribution" in titles
    assert "turnover" in titles


def test_tearsheet_single_asset_stays_2x2():
    equity = np.cumprod(1.0 + np.random.default_rng(0).normal(0.0, 0.01, 100))
    fig = tearsheet(equity)
    assert len(fig.axes) == 4
