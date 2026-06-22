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
    therefore floored with a **returns-scaled** value
    ``|equity|.mean() / MAX_RATIO`` (plus a bare ``eps`` backstop for the
    degenerate all-zero batch), capping the ratio at roughly ``MAX_RATIO``
    in the low-drawdown regime regardless of the return scale.

    The ratio is then passed through a **smooth saturating map**,
    ``MAX_RATIO * tanh(ratio / MAX_RATIO)``, instead of a hard clamp. A
    hard clamp pins the loss to a constant on a near-zero-drawdown batch
    and so zeroes the gradient in exactly the strong-uptrend regime
    training still wants to push on; ``tanh`` is near-linear for
    normal-regime ratios (leaving their numerics unchanged) yet keeps a
    residual, non-zero gradient when the ratio is large. This keeps the
    loss finite and bounded while preserving its sign convention
    (minimizing it maximizes the ratio).

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
        # batch. Scaling by ``|equity|.mean() / MAX_RATIO`` caps the ratio at
        # ~MAX_RATIO in the near-zero-drawdown regime; the bare eps backstop
        # guards the degenerate all-zero-return case.
        floor = equity.abs().mean() / _MAX_RATIO + self.eps
        ratio = annual_return / torch.clamp(max_drawdown, min=floor)

        # Smooth saturating map instead of a hard clamp: tanh is near-linear for
        # normal-regime ratios (numerics unchanged) but keeps a non-zero
        # gradient when the ratio is large, unlike a clamp that zeroes it.
        return -_MAX_RATIO * torch.tanh(ratio / _MAX_RATIO)
