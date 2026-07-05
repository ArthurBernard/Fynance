#!/usr/bin/env python3
# coding: utf-8

""" Factor tear-sheet figures (Alphalens-style).

Composable matplotlib panels for the factor-evaluation metrics in
:mod:`fynance.metrics.factor`, plus a one-call :func:`factor_tearsheet`. Each
panel returns an ``Axes`` (and :func:`factor_tearsheet` a ``Figure``) and never
calls ``show`` — usable headless, in a notebook or an app. Matplotlib is
imported lazily inside each function so ``import fynance`` stays
matplotlib-free.

"""

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np

# Local packages
from fynance.features.horizon import horizon_returns
from fynance.metrics.factor import (
    QuantileResult,
    ic_decay,
    quantile_returns,
    roll_information_coefficient,
)

__all__ = [
    'plot_quantile_returns',
    'plot_ic_series',
    'plot_ic_decay',
    'factor_tearsheet',
]


def _compound(returns: Any) -> Any:
    """ Cumulative compounded growth of a return series (NaN treated as flat). """
    r = np.nan_to_num(np.asarray(returns, dtype=np.float64), nan=0.0)

    return np.cumprod(1.0 + r)


def plot_quantile_returns(result: QuantileResult, ax: Any = None,
                          **kw: Any) -> Any:
    """ Plot the cumulative compounded return of each factor quantile.

    Each bucket's per-bar mean forward return
    (:attr:`~fynance.metrics.factor.QuantileResult.quantile_returns`) is
    compounded into a growth curve, so a monotone fan (bottom bucket lowest, top
    bucket highest) is the signature of a working factor. Returns the ``Axes``.

    Parameters
    ----------
    result : fynance.metrics.factor.QuantileResult
        Output of :func:`fynance.metrics.quantile_returns`.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created when omitted.
    **kw
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    qret = np.asarray(result.quantile_returns, dtype=np.float64)
    Q = result.n_quantiles

    if ax is None:
        _, ax = plt.subplots()

    for q in range(Q):
        ax.plot(_compound(qret[:, q]), lw=1.3, label=f"Q{q + 1}", **kw)
    ax.set_title(f"Quantile cumulative returns (Q={Q})")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    return ax


def plot_ic_series(ic: Any, w_smooth: int = 21, ax: Any = None,
                   **kw: Any) -> Any:
    """ Plot a rolling Information Coefficient with a smoothed overlay.

    Draws the raw IC series (thin) and a trailing moving average over
    ``w_smooth`` bars (bold), plus a zero reference line. Returns the ``Axes``.

    Parameters
    ----------
    ic : array-like
        Rolling IC series, e.g. from
        :func:`fynance.metrics.roll_information_coefficient`.
    w_smooth : int, optional
        Window of the smoothing moving average (default ``21``).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created when omitted.
    **kw
        Forwarded to :meth:`matplotlib.axes.Axes.plot` for the raw series.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    ic = np.asarray(ic, dtype=np.float64)
    T = ic.shape[0]
    w = max(1, min(int(w_smooth), T))

    # Trailing NaN-robust moving average of the IC series.
    smooth = np.full(T, np.nan, dtype=np.float64)
    for t in range(w - 1, T):
        window = ic[t - w + 1:t + 1]
        if np.any(np.isfinite(window)):
            smooth[t] = np.nanmean(window)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(ic, color="#9ecae1", lw=0.9, label="IC", **kw)
    ax.plot(smooth, color="#08519c", lw=1.6, label=f"IC (MA {w})")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("Rolling information coefficient")
    ax.set_ylabel("IC")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    return ax


def plot_ic_decay(decay: Any, horizons: tuple[int, ...], ax: Any = None,
                  **kw: Any) -> Any:
    """ Bar chart of the Information Coefficient across forward horizons.

    Plots the mean IC (e.g. from :func:`fynance.metrics.ic_decay`) at each
    horizon; a signal with genuine short-horizon edge shows a tall first bar
    that decays with the horizon. Returns the ``Axes``.

    Parameters
    ----------
    decay : array-like
        Mean IC per horizon, aligned with ``horizons``.
    horizons : tuple of int
        Forward horizons in bars (the x-axis labels).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created when omitted.
    **kw
        Forwarded to :meth:`matplotlib.axes.Axes.bar`.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    decay = np.asarray(decay, dtype=np.float64)
    x = np.arange(len(horizons))

    if ax is None:
        _, ax = plt.subplots()

    ax.bar(x, decay, color="#3182bd", alpha=0.85, **kw)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_title("IC decay by horizon")
    ax.set_xlabel("Horizon (bars)")
    ax.set_ylabel("Mean IC")
    ax.grid(alpha=0.3, axis="y")

    return ax


def factor_tearsheet(
    factor: Any,
    prices: Any,
    n_quantiles: int = 5,
    horizons: tuple[int, ...] = (1, 5, 10, 21),
    w: int = 63,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
) -> Any:
    """ Build a full Alphalens-style factor tear-sheet figure.

    Composes four panels into a 2x2 ``Figure`` from a factor panel and its price
    panel: quantile cumulative returns, the long-short spread equity, the
    rolling Information Coefficient and the IC decay by horizon. The one-bar
    forward return (built from ``prices``) is the target for the quantile and
    rolling-IC panels; the decay panel uses non-overlapping ``horizons`` returns.

    **Alignment.** ``factor[t]`` is the score known at bar ``t``; the forward
    returns are built from ``prices`` so no future information leaks in.

    Parameters
    ----------
    factor : array-like
        Factor panel ``(T, N)``.
    prices : array-like
        Price panel ``(T, N)`` with the same time index as ``factor``.
    n_quantiles : int, optional
        Number of factor quantiles (default ``5``).
    horizons : tuple of int, optional
        Forward horizons for the IC-decay panel (default ``(1, 5, 10, 21)``).
    w : int, optional
        Trailing window of the rolling IC (default ``63``).
    figsize : tuple of float, optional
        Figure size; defaults to ``(11, 7)``.
    save : str, optional
        If given, the figure is written to this path with
        :meth:`~matplotlib.figure.Figure.savefig`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    import matplotlib.pyplot as plt

    factor = np.asarray(factor, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)

    # One-bar forward return aligned with the factor (drop the last, unlabelled
    # bar): fwd[t] is realized after factor[t] is known.
    fwd = horizon_returns(prices, 1, overlapping=True)
    factor_aligned = factor[:-1]

    qres = quantile_returns(factor_aligned, fwd, n_quantiles=n_quantiles)
    ic = roll_information_coefficient(factor_aligned, fwd, w=w)
    decay = ic_decay(factor, prices, horizons=horizons)

    fig = plt.figure(figsize=figsize or (11, 7))
    gs = fig.add_gridspec(2, 2)

    plot_quantile_returns(qres, ax=fig.add_subplot(gs[0, 0]))

    ax_spread = fig.add_subplot(gs[0, 1])
    ax_spread.plot(_compound(qres.spread), color="#54278f", lw=1.4)
    ax_spread.axhline(1.0, color="k", lw=0.8)
    ax_spread.set_title("Long-short spread equity")
    ax_spread.set_ylabel("Growth of 1")
    ax_spread.grid(alpha=0.3)

    plot_ic_series(ic, ax=fig.add_subplot(gs[1, 0]))
    plot_ic_decay(decay, horizons, ax=fig.add_subplot(gs[1, 1]))

    fig.tight_layout()

    if save is not None:
        fig.savefig(save)

    return fig
