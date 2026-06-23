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
    denominator is therefore floored with a **returns-scaled** value
    ``|r - L|.mean() / MAX_RATIO`` (plus a bare ``eps`` backstop for the
    degenerate all-zero-diff batch), capping the ratio at roughly
    ``MAX_RATIO`` in the low-loss regime regardless of the return scale.

    The ratio is then passed through a **smooth saturating map**,
    ``MAX_RATIO * tanh(ratio / MAX_RATIO)``, instead of a hard clamp. A
    hard clamp pins the loss to a constant on an all-gains batch and so
    zeroes the gradient in exactly the regime training still wants to push
    on; ``tanh`` is near-linear for normal-regime ratios (leaving their
    numerics unchanged) yet keeps a residual, non-zero gradient when the
    ratio is large. This keeps the loss finite and bounded while
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
        y_pred = self._book_return(y_pred)
        diff = y_pred - self.threshold
        gains = torch.relu(diff).mean()
        losses = torch.relu(-diff).mean()
        # Returns-scaled floor: a fixed absolute eps is dimensionally wrong for
        # O(|r - L|) losses and lets the ratio explode on an all-gains batch.
        # Scaling by ``|r - L|.mean() / MAX_RATIO`` caps the ratio at ~MAX_RATIO
        # in the low-loss regime; the bare eps backstop guards the degenerate
        # all-zero-diff case.
        floor = diff.abs().mean() / _MAX_RATIO + self.eps
        ratio = gains / torch.clamp(losses, min=floor)

        # Smooth saturating map instead of a hard clamp: tanh is near-linear for
        # normal-regime ratios (numerics unchanged) but keeps a non-zero
        # gradient when the ratio is large, unlike a clamp that zeroes it.
        return -_MAX_RATIO * torch.tanh(ratio / _MAX_RATIO)
