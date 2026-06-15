#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Some tools to backtest strategies

.. currentmodule:: fynance.backtest

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   backtest.tools
   backtest.plot_object

"""

from . import (
    backtest_neural_net,
    cost,
    dynamic_plot_backtest,
    engine,
    plot_backtest,
    plot_tools,
    print_stats,
    result,
)
from .backtest_neural_net import *
from .cost import *
from .dynamic_plot_backtest import *
from .engine import *
from .plot_backtest import *
from .plot_tools import *
from .print_stats import *
from .result import *

__all__ = print_stats.__all__
__all__ += cost.__all__
__all__ += engine.__all__
__all__ += result.__all__
__all__ += plot_tools.__all__
__all__ += plot_backtest.__all__
__all__ += dynamic_plot_backtest.__all__
__all__ += backtest_neural_net.__all__
