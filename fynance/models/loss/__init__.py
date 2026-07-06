#!/usr/bin/env python3
# coding: utf-8

""" Differentiable financial loss functions for PyTorch training.

All losses are pure PyTorch — no NumPy conversion — so gradients flow
through them without breaking the autograd graph. They are designed as
drop-in replacements for standard PyTorch criterions and integrate
directly with :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`.

Main entry points
-----------------
- :class:`SharpeLoss` — negative Sharpe ratio (minimizes → maximizes
  risk-adjusted return).
- :class:`SortinoLoss` — negative Sortino ratio (downside-deviation
  proxy; only penalizes negative excess returns).
- :class:`DirectionalAccuracyLoss` — sigmoid surrogate for directional
  accuracy (differentiable approximation of sign-prediction rate).
- :class:`CalmarLoss` — negative Calmar ratio (return per unit of
  maximum drawdown).
- :class:`OmegaLoss` — negative Omega ratio (expected gains over
  expected losses relative to a threshold).
- :class:`RankingLoss` — differentiable cross-sectional long-short
  ranking objective for a panel of assets.
- :class:`HybridLoss` — convex combination of two losses with a fixed or
  learnable weight.
- :class:`PinballLoss` — asymmetric quantile ("pinball") loss for
  multi-quantile regression (see
  :class:`~fynance.models.quantile.QuantileModel`).

Notes
-----
Each loss is a **training proxy**: its scalar value is not numerically
comparable to the corresponding evaluation metric in
:mod:`fynance.metrics`. Use those metrics for out-of-sample
reporting; use these losses only inside training loops.

"""

from __future__ import annotations

# Local packages
from ._base import BaseLoss
from .calmar import CalmarLoss
from .directional import DirectionalAccuracyLoss
from .hybrid import HybridLoss
from .omega import OmegaLoss
from .pinball import PinballLoss
from .ranking import RankingLoss
from .sharpe import SharpeLoss
from .sortino import SortinoLoss

__all__ = ['BaseLoss', 'CalmarLoss', 'DirectionalAccuracyLoss', 'HybridLoss',
           'OmegaLoss', 'PinballLoss', 'RankingLoss', 'SharpeLoss', 'SortinoLoss']
