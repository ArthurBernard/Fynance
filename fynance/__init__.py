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
portfolio       --- Portfolio allocation & sizing
backtest        --- Backtest strategy tools
estimator       --- Parameter estimation (Cython ARMA/GARCH)
features        --- Features extraction
models          --- Econometric and Neural Network models (using PyTorch)

Utility tools
-------------
_exceptions --- Fynance exceptions
tests       --- Run fynance unittests
_wrappers   --- Fynance wrapper functions

API stability policy (1.x series)
---------------------------------
The symbols re-exported below from :mod:`fynance.models`,
:mod:`fynance.portfolio.allocation`, :mod:`fynance.features` and
:mod:`fynance.estimator` form the **public, stable API** for the 1.x
release line. Within 1.x:

- public function and class signatures are frozen — no removals, no
  backward-incompatible signature changes;
- behavioural changes that would break user code go through one
  release of :class:`DeprecationWarning` before becoming the new
  default (see ``CONTRIBUTING.md``);
- :mod:`fynance.backtest` and :mod:`fynance.models` *internal* helpers
  (names prefixed with ``_``) remain free to evolve.

Breaking changes are reserved for the 2.x line and tracked in
``CHANGELOG.md``.

"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fynance")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ['__version__']

import sys as _sys

from .backtest import *
from .estimator import *
from .features import *
from .metrics import *
from .models import *
from .plot import *
from .portfolio import *
from .strategy import *

# Aggregate each subpackage's public surface. Use ``sys.modules`` rather than the
# package attributes, which a star import may have shadowed with a name that
# collides with a submodule (e.g. the ``backtest`` engine function).
for _name in ("models", "estimator", "features", "metrics", "plot", "backtest", "portfolio", "strategy"):
    __all__ += _sys.modules[f"{__name__}.{_name}"].__all__

# Restore the subpackage attribute shadowed by such a collision so that
# ``fynance.backtest`` resolves to the package (``fynance.backtest.backtest``
# stays the engine function).
backtest = _sys.modules[__name__ + ".backtest"]

del _sys, _name
