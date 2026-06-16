#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Vectorized backtesting engine and result.

.. currentmodule:: fynance.backtest

The vectorized engine (:func:`backtest`), its :class:`BacktestResult`, cost
models and the text performance summary. Plotting lives in :mod:`fynance.plot`
(:func:`~fynance.plot.tearsheet`). The legacy live-training plot objects
(``PlotBackTest`` / ``DynaPlotBackTest`` / ``display_perf``) remain as on-demand
submodules used by :class:`~fynance.models.rolling.RollMultiLayerPerceptron`, but
are no longer part of the eager public surface — importing :mod:`fynance` no
longer pulls matplotlib.

"""

from . import backtest_neural_net, cost, engine, print_stats, result
from .backtest_neural_net import *
from .cost import *
from .engine import *
from .print_stats import *
from .result import *

__all__ = print_stats.__all__
__all__ += cost.__all__
__all__ += engine.__all__
__all__ += result.__all__
__all__ += backtest_neural_net.__all__
