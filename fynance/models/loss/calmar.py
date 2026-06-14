#!/usr/bin/env python3
# coding: utf-8

""" Differentiable Calmar-ratio loss. """

from __future__ import annotations

# Third-party packages
import torch

# Local packages
from ._base import BaseLoss

__all__ = ['CalmarLoss']


class CalmarLoss(BaseLoss):
    r""" Negative Calmar ratio as a differentiable loss.

    Calmar = annualized return / maximum drawdown. Minimizing this loss
    maximizes return per unit of worst peak-to-trough loss. The maximum
    drawdown is computed differentiably from the cumulative return path
    via :func:`torch.cummax`.

    Parameters are inherited from :class:`BaseLoss` (``period``, ``eps``).

    """

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Compute the negative Calmar ratio (scalar). """
        self._check_tensor(y_pred)
        equity = torch.cumsum(y_pred, dim=0)
        running_max, _ = torch.cummax(equity, dim=0)
        max_drawdown = (running_max - equity).max()
        annual_return = y_pred.mean() * self.period

        return -(annual_return / (max_drawdown + self.eps))
