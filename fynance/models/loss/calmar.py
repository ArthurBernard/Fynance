#!/usr/bin/env python3
# coding: utf-8

""" Differentiable Calmar-ratio loss. """

from __future__ import annotations

# Third-party packages
import torch

# Local packages
from ._base import MAX_RATIO as _MAX_RATIO
from ._base import BaseLoss

__all__ = ['CalmarLoss']


class CalmarLoss(BaseLoss):
    r""" Negative Calmar ratio as a differentiable loss.

    Calmar = annualized return / maximum drawdown. Minimizing this loss
    maximizes return per unit of worst peak-to-trough loss. The maximum
    drawdown is computed differentiably from the cumulative return path
    via :func:`torch.cummax`.

    Parameters are inherited from :class:`BaseLoss` (``period``, ``eps``).

    Notes
    -----
    Drawdowns are :math:`O(\text{returns})`, so a fixed absolute ``eps``
    (e.g. ``1e-8``) is dimensionally wrong: on a low- or zero-drawdown
    batch the ratio would explode and dominate gradients. The drawdown is
    therefore floored with a **returns-scaled** epsilon,
    ``eps * |equity|.mean()``, keeping the loss finite and bounded while
    preserving its sign convention (minimizing it maximizes the ratio).

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
        # Returns-scaled floor: a fixed absolute eps is dimensionally wrong for
        # an O(returns) drawdown and lets the ratio explode on a low-drawdown
        # batch. Scaling by the equity magnitude keeps the floor on the right
        # scale; the bare eps backstop guards the degenerate all-zero-return
        # case and the final clamp bounds the magnitude when the drawdown is
        # near zero (so the loss stays finite with well-scaled gradients).
        floor = self.eps * equity.abs().mean() + self.eps
        ratio = annual_return / torch.clamp(max_drawdown, min=floor)

        return -torch.clamp(ratio, min=-_MAX_RATIO, max=_MAX_RATIO)
