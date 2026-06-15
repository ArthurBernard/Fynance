#!/usr/bin/env python3
# coding: utf-8

""" Equity and drawdown figures. """

from __future__ import annotations

# Built-in packages
from typing import Any

# Local packages
from fynance.plot._helpers import as_equity, drawdown_curve

__all__ = ['plot_equity', 'plot_drawdown']


def plot_equity(result: Any, ax: Any = None) -> Any:
    """ Plot an equity curve. Returns the matplotlib ``Axes``. """
    import matplotlib.pyplot as plt

    equity, index = as_equity(result)
    x = range(len(equity)) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, equity, color="#2c7fb8", lw=1.5)
    ax.set_title("Equity curve")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)

    return ax


def plot_drawdown(result: Any, ax: Any = None) -> Any:
    """ Plot the underwater drawdown curve. Returns the ``Axes``. """
    import matplotlib.pyplot as plt

    equity, index = as_equity(result)
    dd = drawdown_curve(equity)
    x = range(len(dd)) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    ax.fill_between(x, dd, 0.0, color="#d7301f", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)

    return ax
