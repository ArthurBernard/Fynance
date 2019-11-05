#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-15 12:50:12
# @Last modified by: ArthurBernard
# @Last modified time: 2019-11-05 16:55:36

"""
Fynance : A Python package for quant financial research
=======================================================

Documentation is available at
https://fynance.readthedocs.io/en/latest/index.html.

Contents
--------
Fynance is a python/cython project that includes several machine learning,
econometric and statistical subpackages specialy adapted for financial
analysis, portfolio allocation, and backtest trading strategies.

Subpackages
-----------
algorithms      --- Financial algorithms
backtest        --- Backtest strategy tools
features        --- Features extraction
models          --- Econometric and Neural Network models (using PyTorch)
neural_networks --- Neural network models (using Keras)

Utility tools
-------------
_exceptions --- Fynance exceptions
tests       --- Run fynance unittests
_wrappers   --- Fynance wrapper functions

"""

from .version import version as __version__

__all__ = ['__version__']

from .models import *
from .estimator import *
from .features import *
from .neural_networks import *
from .backtest import *
from .algorithms import *

__all__ += models.__all__
__all__ += estimator.__all__
__all__ += features.__all__
__all__ += neural_networks.__all__
__all__ += backtest.__all__
__all__ += algorithms.__all__
