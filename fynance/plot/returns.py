#!/usr/bin/env python3
# coding: utf-8

""" Return-distribution and rolling-metric figures. """

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np

# Local packages
from fynance.metrics.ratios import roll_sharpe
from fynance.plot._helpers import as_equity

__all__ = ['plot_returns_hist', 'plot_rolling_sharpe']


def plot_returns_hist(result: Any, bins: int = 50, ax: Any = None) -> Any:
    """ Histogram of per-step returns. Returns the ``Axes``. """
    import matplotlib.pyplot as plt

    equity, _ = as_equity(result)
    returns = equity[1:] / equity[:-1] - 1.0

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(returns, bins=bins, color="#2c7fb8", alpha=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_title("Return distribution")
    ax.set_xlabel("Return")
    ax.grid(alpha=0.3)

    return ax


def plot_rolling_sharpe(result: Any, window: int = 252, ax: Any = None) -> Any:
    """ Rolling Sharpe ratio of the equity curve. Returns the ``Axes``. """
    import matplotlib.pyplot as plt

    equity, index = as_equity(result)
    w = min(window, max(2, len(equity) - 1))
    rs = np.asarray(roll_sharpe(equity, w=w))
    x = range(len(rs)) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, rs, color="#238b45", lw=1.2)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title(f"Rolling Sharpe (w={w})")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3)

    return ax
