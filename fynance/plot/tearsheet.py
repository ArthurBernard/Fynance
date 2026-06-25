#!/usr/bin/env python3
# coding: utf-8

""" One-call performance report (the API the notebook and UI both use). """

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np

# Local packages
from fynance.plot._helpers import as_equity
from fynance.plot.attribution import plot_contribution, plot_turnover
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


def tearsheet(result: Any, period: int = 252, figsize: tuple = (11, 7), *,
              base: float | None = None, logy: bool | str = "auto") -> Any:
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
    base : float, optional
        Rescale the equity panel to start at ``base`` (e.g. ``100`` for the
        familiar base-100 reading); display only, see :func:`plot_equity`.
    logy : bool or {"auto"}, default "auto"
        Log y-axis policy for the equity panel; ``"auto"`` switches to log on
        wide-amplitude curves, see :func:`plot_equity`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    import matplotlib.pyplot as plt

    # A multi-asset book carries per-asset attribution: add a contribution and a
    # turnover panel below the core 2x2 report (single-asset stays a 2x2).
    asset_gross = getattr(result, "asset_gross_returns", None)
    positions = getattr(result, "positions", None)
    index = getattr(result, "index", None)
    is_book = (
        asset_gross is not None
        and np.asarray(asset_gross).ndim == 2
        and np.asarray(asset_gross).shape[1] > 1
    )

    n_rows = 3 if is_book else 2
    fig = plt.figure(figsize=(figsize[0], figsize[1] * (1.4 if is_book else 1.0)))
    gs = fig.add_gridspec(n_rows, 2)

    plot_equity(result, ax=fig.add_subplot(gs[0, 0]), base=base, logy=logy)
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

    if is_book:
        plot_contribution(asset_gross, index=index, ax=fig.add_subplot(gs[2, 0]))
        if positions is not None and np.asarray(positions).ndim == 2:
            plot_turnover(positions, index=index, ax=fig.add_subplot(gs[2, 1]))

    fig.tight_layout()

    return fig


def tearsheet_text(result: Any, period: int = 252) -> str:
    """ Plain-text performance summary (for notebooks / CLI). """
    stats = _summary(result, period)

    return "\n".join(f"{k:<20s} {v:>12.4f}" for k, v in stats.items())
