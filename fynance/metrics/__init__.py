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
from .drawdown import *
from .ratios import *
from .returns import *
from .returns import perf_strat, returns_strat  # noqa: F401

# Aggregate __all__ via sys.modules (the star imports above rebind names that
# collide with a submodule, e.g. the ``drawdown`` function vs the module).
__all__ = []
for _m in ("returns", "ratios", "drawdown"):
    __all__ += _sys.modules[f"{__name__}.{_m}"].__all__
__all__ += ['perf_strat', 'returns_strat']

del _sys, _m
