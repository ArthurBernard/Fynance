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
    dynamic_plot_backtest,
    plot_backtest,
    plot_tools,
    print_stats,
)
from .backtest_neural_net import *
from .dynamic_plot_backtest import *
from .plot_backtest import *
from .plot_tools import *
from .print_stats import *

__all__ = print_stats.__all__
__all__ += plot_tools.__all__
__all__ += plot_backtest.__all__
__all__ += dynamic_plot_backtest.__all__
__all__ += backtest_neural_net.__all__
