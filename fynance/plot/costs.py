#!/usr/bin/env python3
# coding: utf-8

""" Transaction-cost decomposition figures. """

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np

__all__ = ['plot_cost_decomposition']


def plot_cost_decomposition(cost_components: dict[str, Any], index: Any = None,
                            ax: Any = None) -> Any:
    """ Stack the cumulative cost components as a share of capital.

    ``cost_components`` maps a component name (e.g. ``"transaction"``,
    ``"market_impact"``) to its per-step cost series; each is accumulated and
    stacked, so the panel reads as cumulative fees in **% of capital** broken
    down by source — the detail a single aggregate ``total_cost`` hides. Returns
    the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    names = list(cost_components)
    cumulative = [
        np.nancumsum(np.asarray(cost_components[k], dtype=np.float64)) * 100.0
        for k in names
    ]
    x = range(cumulative[0].shape[0]) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    ax.stackplot(x, *cumulative, labels=names, alpha=0.8)
    ax.set_title("Cumulative fees")
    ax.set_ylabel("Cumulative cost (% of capital)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    return ax
