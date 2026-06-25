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
from .correlation import *
from .drawdown import *
from .ratios import *
from .returns import *
from .returns import perf_strat, returns_strat  # noqa: F401
from .summary import METRICS, summary  # noqa: F401
from .trading import *

# Aggregate __all__ via sys.modules (the star imports above rebind names that
# collide with a submodule, e.g. the ``drawdown`` function vs the module).
# ``information_coefficient`` and the trade-profile metrics (``sign_changes`` /
# ``trades_per_year``) are exported here but intentionally NOT added to the
# ``METRICS`` registry: that registry maps a name to a callable taking a single
# equity/price curve (see ``summary``), whereas the IC takes an aligned
# (pred, real) pair and the trade-profile metrics take a position series, so
# neither fits that single-series contract.
__all__ = []
for _m in ("correlation", "returns", "ratios", "drawdown", "trading"):
    __all__ += _sys.modules[f"{__name__}.{_m}"].__all__
__all__ += ['perf_strat', 'returns_strat', 'METRICS', 'summary']

del _sys, _m
