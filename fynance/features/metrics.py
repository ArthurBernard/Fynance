#!/usr/bin/env python3
# coding: utf-8

""" Performance and risk metrics for financial analysis.

Compute risk-adjusted returns, drawdown statistics and other summary
indicators commonly used to evaluate strategies and portfolios. All
functions accept a 1-D or 2-D array of prices/returns and return the
metric along the time axis.

Main entry points
-----------------
- :func:`annual_return`, :func:`annual_volatility` — annualized
  return and volatility from a price series.
- :func:`sharpe`, :func:`calmar`, :func:`diversified_ratio` —
  risk-adjusted performance ratios.
- :func:`mdd`, :func:`drawdown` — maximum drawdown and drawdown path.
- :func:`z_score`, :func:`accuracy` — statistical helpers.

"""

from fynance.features.drawdown import *
from fynance.features.ratios import *
from fynance.features.returns import *

# non-__all__ public helpers still imported directly by some callers/tests
from fynance.features.returns import perf_strat, returns_strat  # noqa: F401
from fynance.features.stats import *

__all__ = ['accuracy', 'annual_return', 'annual_volatility', 'calmar', 'directional_accuracy', 'diversified_ratio', 'drawdown', 'mad', 'mdd', 'roll_annual_return', 'roll_annual_volatility', 'roll_calmar', 'roll_drawdown', 'roll_mad', 'roll_mdd', 'roll_sharpe', 'roll_z_score', 'sharpe', 'sortino', 'perf_index', 'perf_returns', 'z_score']
