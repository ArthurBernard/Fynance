#!/usr/bin/env python3
# coding: utf-8

""" Equity and drawdown figures. """

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.plot._helpers import as_equity, drawdown_curve

__all__ = ['plot_equity', 'plot_drawdown']

# Auto log-scale kicks in once the equity spans more than this multiplicative
# range (max / min): a linear axis then crushes the early trajectory and makes
# drawdowns visually incomparable across time.
_LOG_Y_RATIO = 5.0


def _use_log_y(equity: NDArray, logy: bool | str) -> bool:
    """ Resolve the ``logy`` policy into a concrete log/linear choice. """
    positive = bool(equity.size) and bool(np.all(equity > 0.0))

    if logy == "auto":
        if not positive:

            return False

        return float(equity.max() / equity.min()) > _LOG_Y_RATIO

    return bool(logy) and positive


def plot_equity(result: Any, ax: Any = None, *, base: float | None = None,
                logy: bool | str = "auto") -> Any:
    """ Plot an equity curve. Returns the matplotlib ``Axes``.

    Parameters
    ----------
    result : BacktestResult, PriceSeries or array-like
        The strategy result (or a raw equity curve).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created when omitted.
    base : float, optional
        Rescale the curve to start at ``base`` (display only). The default
        equity starts at the capital (``1.0``); ``base=100`` gives the familiar
        base-100 reading without touching the underlying numbers.
    logy : bool or {"auto"}, default "auto"
        Use a logarithmic y-axis. ``"auto"`` switches to log only when the curve
        spans more than a 5x range (and stays strictly positive), so a x3-x30
        trajectory stays readable while a flat curve is left linear.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    equity, index = as_equity(result)

    if (base is not None and equity.size and np.isfinite(equity[0])
            and equity[0] != 0.0):
        equity = equity * (base / equity[0])

    x = range(len(equity)) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, equity, color="#2c7fb8", lw=1.5)
    ax.set_title("Equity curve")
    ax.set_ylabel("Equity" if base is None else f"Equity (base {base:g})")
    ax.grid(alpha=0.3)

    if _use_log_y(equity, logy):
        ax.set_yscale("log")

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
