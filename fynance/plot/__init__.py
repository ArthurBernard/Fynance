#!/usr/bin/env python3
# coding: utf-8

""" Reporting / plotting layer.

.. currentmodule:: fynance.plot

Small, composable matplotlib figures (each returns an ``Axes``/``Figure``,
never calls ``show``) plus a one-call :func:`tearsheet` report — the reporting
API the notebook workflow and the optional Streamlit app both build on.

"""

# Local packages
from .attribution import plot_contribution, plot_turnover
from .costs import plot_cost_decomposition
from .equity import plot_drawdown, plot_equity
from .exposure import plot_exposure
from .factor import (
    factor_tearsheet,
    plot_ic_decay,
    plot_ic_series,
    plot_quantile_returns,
)
from .returns import plot_returns_hist, plot_rolling_sharpe
from .tearsheet import tearsheet, tearsheet_text

__all__ = [
    'plot_equity',
    'plot_drawdown',
    'plot_returns_hist',
    'plot_rolling_sharpe',
    'plot_contribution',
    'plot_turnover',
    'plot_cost_decomposition',
    'plot_exposure',
    'plot_quantile_returns',
    'plot_ic_series',
    'plot_ic_decay',
    'factor_tearsheet',
    'tearsheet',
    'tearsheet_text',
]
