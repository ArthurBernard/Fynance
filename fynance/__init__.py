#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 

Fynance package is financial tools in python.

.. automodule:: fynance
   :members:

"""

from . import models
from .models import *
from . import estimator
from .estimator import *
from . import tools
from .tools import *
from . import neural_networks
from .neural_networks import *
from . import backtest
from .backtest import *

#__version__ = '1.0.4'

__all__ = models.__all__
__all__ += estimator.__all__
__all__ += tools.__all__
__all__ += neural_networks.__all__
__all__ += backtest.__all__

#__all__ = [
#    'get_parameters', 'MA', 'ARMA', 'ARMA_GARCH', 'ARMAX_GARCH', # models
#    'estimator', # not ready
#    'sharpe', 'roll_sharpe', 'sma', 'wma', 'ema', 'smstd', 'wmstd', 'emstd', 
#    'z_score', 'rsi', 'bollinger_band', 'hma', 'macd_line', 'signal_line', 
#    'macd_hist', 'mdd', 'calmar', 'roll_mdd', 'roll_calmar', 'drawdown', # tools 
#    'RollNeuralNet', # neural_network
#]