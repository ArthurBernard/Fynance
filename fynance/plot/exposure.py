#!/usr/bin/env python3
# coding: utf-8

""" Gross / net exposure figure.

Composable matplotlib panel for the book-level exposure metrics in
:mod:`fynance.metrics.trading`. Matplotlib is imported lazily inside the
function so ``import fynance`` stays matplotlib-free.

"""

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np

# Local packages
from fynance.metrics.trading import gross_exposure, net_exposure

__all__ = ['plot_exposure']


def plot_exposure(W: Any, ax: Any = None, **kw: Any) -> Any:
    """ Plot the book's gross and net exposure over time.

    ``W`` is the weight/position book, shape ``(T,)`` (promoted to ``(T, 1)``)
    or ``(T, N)``. Gross exposure
    (:func:`~fynance.metrics.trading.gross_exposure`, :math:`\\sum_i |w_i|`)
    reads as total book leverage; net exposure
    (:func:`~fynance.metrics.trading.net_exposure`, :math:`\\sum_i w_i`) reads
    as the long/short bias — plotting both together shows at a glance whether
    a high-leverage book is directionally hedged or one-sided. Returns the
    matplotlib ``Axes``.

    Parameters
    ----------
    W : array_like
        Weights held at each step, shape ``(T,)`` or ``(T, N)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    **kw
        ``index`` (array_like, optional) overrides the x-axis (defaults to
        ``range(T)``); any other keyword is forwarded to both ``ax.plot``
        calls (e.g. ``lw``, ``alpha``).

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    index = kw.pop("index", None)
    w = np.asarray(W, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    gross = gross_exposure(w)
    net = net_exposure(w)
    x = range(gross.shape[0]) if index is None else index

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, gross, color="#2c7fb8", lw=1.2, label="gross", **kw)
    ax.plot(x, net, color="#d7301f", lw=1.2, label="net", **kw)
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_title("Book exposure")
    ax.set_ylabel("Exposure")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    return ax
