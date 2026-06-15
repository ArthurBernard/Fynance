#!/usr/bin/env python3
# coding: utf-8

""" One-call performance report (the API the notebook and UI both use). """

from __future__ import annotations

# Built-in packages
from typing import Any

# Local packages
from fynance.plot._helpers import as_equity
from fynance.plot.equity import plot_drawdown, plot_equity
from fynance.plot.returns import plot_rolling_sharpe

__all__ = ['tearsheet', 'tearsheet_text']


def _summary(result: Any, period: int) -> dict[str, float]:
    """ Resolve a metrics summary from a result/series/array. """
    if hasattr(result, "summary"):

        return result.summary(period=period)

    from fynance.metrics import summary

    equity, _ = as_equity(result)

    return summary(equity, period=period)


def tearsheet(result: Any, period: int = 252, figsize: tuple = (11, 7)) -> Any:
    """ Build a full performance report figure.

    Composes the equity curve, drawdown, rolling Sharpe, return distribution
    and a metrics table into one matplotlib ``Figure`` (works headless, embeds
    in a notebook or a Streamlit app).

    Parameters
    ----------
    result : BacktestResult, PriceSeries or array-like
        The strategy result (or an equity curve).
    period : int
        Annualization factor.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure

    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)

    plot_equity(result, ax=fig.add_subplot(gs[0, 0]))
    plot_drawdown(result, ax=fig.add_subplot(gs[0, 1]))
    plot_rolling_sharpe(result, window=period, ax=fig.add_subplot(gs[1, 0]))

    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")
    stats = _summary(result, period)
    rows = [[k, f"{v:.4f}"] for k, v in stats.items()]
    table = ax_table.table(cellText=rows, colLabels=["metric", "value"],
                           loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax_table.set_title("Summary")

    fig.tight_layout()

    return fig


def tearsheet_text(result: Any, period: int = 252) -> str:
    """ Plain-text performance summary (for notebooks / CLI). """
    stats = _summary(result, period)

    return "\n".join(f"{k:<20s} {v:>12.4f}" for k, v in stats.items())
