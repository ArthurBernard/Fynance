#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-15 12:50:12
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-12 17:56:06

"""

Fynance package is deep learning tools for financial analysis with Python.

.. automodule:: fynance
   :members:

"""

from .models import *
from .estimator import *
from .tools import *
from .neural_networks import *
from .backtest import *
from .algorithms import *

# __version__ = '1.0.4'

__all__ = models.__all__
__all__ += estimator.__all__
__all__ += tools.__all__
__all__ += neural_networks.__all__
__all__ += backtest.__all__
__all__ += algorithms.__all__

# __all__ = [
#    'get_parameters', 'MA', 'ARMA', 'ARMA_GARCH', 'ARMAX_GARCH', # models
#    'estimator', # not ready
#    'sharpe', 'roll_sharpe', 'sma', 'wma', 'ema', 'smstd', 'wmstd', 'emstd',
#    'z_score', 'rsi', 'bollinger_band', 'hma', 'macd_line', 'signal_line',
#    'macd_hist', 'mdd', 'calmar', 'roll_mdd', 'roll_calmar', 'drawdown',
#    'RollNeuralNet', # neural_network, # tools,
# ]
