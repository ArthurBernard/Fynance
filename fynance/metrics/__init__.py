#!/usr/bin/env python3
# coding: utf-8

""" Performance / evaluation metrics.

.. currentmodule:: fynance.metrics

Risk-adjusted ratios, return and drawdown metrics for evaluating a strategy
(out-of-sample Sharpe, Sortino, Calmar, max drawdown, ...). Separated from
:mod:`fynance.features`: a metric *evaluates* a series, it is not a feature.

"""

# Built-in packages
import sys as _sys

# Local packages
from .benchmark import *
from .correlation import *
from .drawdown import *
from .factor import *
from .ratios import *
from .returns import *
from .returns import perf_strat, returns_strat  # noqa: F401
from .summary import METRICS, summary  # noqa: F401
from .trading import *

# Aggregate __all__ via sys.modules (the star imports above rebind names that
# collide with a submodule, e.g. the ``drawdown`` function vs the module).
# ``information_coefficient``, the factor-analysis helpers (``quantile_returns``,
# ``roll_information_coefficient``, ``ic_decay``, ``ic_summary``,
# ``factor_rank_autocorr``), the trade-profile metrics (``sign_changes`` /
# ``trades_per_year``) and the benchmark-relative metrics (``beta``, ``alpha``,
# ``tracking_error``, ``information_ratio``, ``capture_ratio``,
# ``benchmark_summary``, ``roll_beta_benchmark``) are exported here but
# intentionally NOT added to the ``METRICS`` registry: that registry maps a name
# to a callable taking a single equity/price curve (see ``summary``), whereas
# the IC and factor helpers take an aligned (pred, real) / (factor, fwd) pair,
# the trade-profile metrics take a position series, and the benchmark metrics
# take an aligned (strategy, benchmark) pair, so none fit that single-series
# contract.
__all__ = []
for _m in ("benchmark", "correlation", "returns", "ratios", "drawdown", "factor", "trading"):
    __all__ += _sys.modules[f"{__name__}.{_m}"].__all__
__all__ += ['perf_strat', 'returns_strat', 'METRICS', 'summary']

del _sys, _m
