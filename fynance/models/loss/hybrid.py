#!/usr/bin/env python3
# coding: utf-8

""" Multi-objective hybrid loss combining two financial losses. """

from __future__ import annotations

# Built-in packages
import math

# Third-party packages
import torch

# Local packages
from ._base import BaseLoss

__all__ = ['HybridLoss']


class HybridLoss(BaseLoss):
    r""" Convex combination of two losses: :math:`\alpha L_a + (1-\alpha) L_b`.

    Combines two differentiable objectives (e.g. :class:`SharpeLoss` and
    :class:`DirectionalAccuracyLoss`) with a weight ``alpha``. The weight can
    be fixed or made **learnable** (an ``nn.Parameter`` passed through a sigmoid
    so it stays in ``(0, 1)`` and is optimized jointly with the model).

    Parameters
    ----------
    loss_a, loss_b : BaseLoss
        The two component losses (any callables ``(y_pred, y_true) -> scalar``).
    alpha : float, optional
        Weight of ``loss_a`` in ``[0, 1]``. Default 0.5.
    learnable : bool, optional
        If True, ``alpha`` becomes a learnable parameter. Default False.
    **kwargs
        Forwarded to :class:`BaseLoss`.

    """

    def __init__(self, loss_a, loss_b, alpha: float = 0.5,
                 learnable: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.learnable = learnable
        if learnable:
            a = min(max(alpha, 1e-4), 1 - 1e-4)
            self._alpha_raw = torch.nn.Parameter(
                torch.tensor(math.log(a / (1 - a)))
            )
        else:
            self.alpha = alpha

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Compute the weighted sum of the two losses (scalar). """
        a = torch.sigmoid(self._alpha_raw) if self.learnable else self.alpha

        return a * self.loss_a(y_pred, y_true) + (1 - a) * self.loss_b(y_pred, y_true)
