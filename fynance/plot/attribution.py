#!/usr/bin/env python3
# coding: utf-8

""" Per-asset attribution figures for a multi-asset book. """

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np

__all__ = ['plot_contribution', 'plot_turnover']


def _as_book(arr: Any) -> np.ndarray:
    """ Coerce to a 2-D ``(T, N)`` book (a 1-D series becomes a single column). """
    a = np.asarray(arr, dtype=np.float64)

    return a if a.ndim == 2 else a.reshape(a.shape[0], -1)


def plot_contribution(asset_returns: Any, index: Any = None, ax: Any = None) -> Any:
    """ Plot the cumulative per-asset gross contribution of a book.

    ``asset_returns`` is the per-asset gross return book ``(T, N)`` (each column
    is one asset's contribution; together they sum to the book gross return). The
    cumulative sum per asset shows how much each asset added to the book over
    time. Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    cum = np.cumsum(_as_book(asset_returns), axis=0)
    x = range(cum.shape[0]) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    for i in range(cum.shape[1]):
        ax.plot(x, cum[:, i], lw=1.2, label=f"asset {i}")
    ax.set_title("Per-asset contribution (cumulative gross)")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    return ax


def plot_turnover(positions: Any, index: Any = None, ax: Any = None) -> Any:
    """ Plot the per-asset turnover ``|Δ position|`` of a book.

    ``positions`` is the position book ``(T, N)``; the turnover at each step is
    the absolute change in position per asset (the first step charges entry from
    flat, matching the cost model). Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    pos = _as_book(positions)
    turn = np.abs(np.diff(pos, axis=0, prepend=np.zeros((1, pos.shape[1]))))
    x = range(turn.shape[0]) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    for i in range(turn.shape[1]):
        ax.plot(x, turn[:, i], lw=1.0, label=f"asset {i}")
    ax.set_title("Per-asset turnover")
    ax.set_ylabel("|Δ position|")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    return ax
