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

from . import print_stats
from .print_stats import *
from . import plot_tools
from .plot_tools import *
from . import plot_backtest
from .plot_backtest import *
from . import dynamic_plot_backtest
from .dynamic_plot_backtest import *

__all__ = print_stats.__all__
__all__ += plot_tools.__all__
__all__ += plot_backtest.__all__
__all__ += dynamic_plot_backtest.__all__