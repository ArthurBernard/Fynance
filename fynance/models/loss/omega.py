#!/usr/bin/env python3
# coding: utf-8

""" Differentiable Omega-ratio loss. """

from __future__ import annotations

# Third-party packages
import torch

# Local packages
from ._base import MAX_RATIO as _MAX_RATIO
from ._base import BaseLoss

__all__ = ['OmegaLoss']


class OmegaLoss(BaseLoss):
    r""" Negative Omega ratio as a differentiable loss.

    :math:`\Omega = \frac{E[\max(r - L, 0)]}{E[\max(L - r, 0)] + \varepsilon}`,
    the ratio of expected gains to expected losses relative to a threshold
    ``L``. Fully differentiable through :func:`torch.relu`. Minimizing the
    loss maximizes the Omega ratio.

    Notes
    -----
    Both gains and losses are :math:`O(|r - L|)`, so a fixed absolute
    ``eps`` is dimensionally wrong: on an all-gains batch (zero losses) the
    ratio would explode (e.g. ``-1e6``) and dominate gradients. The
    denominator is therefore floored with a **returns-scaled** epsilon,
    ``eps * |r - L|.mean()``, keeping the loss finite and bounded while
    preserving the sign convention (minimizing it maximizes the ratio).

    Parameters
    ----------
    threshold : float, optional
        Return threshold ``L`` separating gains from losses. Default 0.
    **kwargs
        Forwarded to :class:`BaseLoss` (``rf``, ``period``, ``eps``).

    """

    def __init__(self, threshold: float = 0., **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Compute the negative Omega ratio (scalar). """
        self._check_tensor(y_pred)
        diff = y_pred - self.threshold
        gains = torch.relu(diff).mean()
        losses = torch.relu(-diff).mean()
        # Returns-scaled floor: a fixed absolute eps is dimensionally wrong for
        # O(|r - L|) losses and lets the ratio explode on an all-gains batch.
        # The bare eps backstop guards the degenerate all-zero-diff case and
        # the final clamp bounds the magnitude when losses are near zero.
        floor = self.eps * diff.abs().mean() + self.eps
        ratio = gains / torch.clamp(losses, min=floor)

        return -torch.clamp(ratio, min=-_MAX_RATIO, max=_MAX_RATIO)
