#!/usr/bin/env python3
# coding: utf-8

""" Differentiable directional accuracy loss for PyTorch training loops. """

from __future__ import annotations

# Third-party packages
import torch

# Local packages
from ._base import BaseLoss

__all__ = ['DirectionalAccuracyLoss']


class DirectionalAccuracyLoss(BaseLoss):
    r""" Negative directional accuracy as a differentiable surrogate loss.

    The true directional accuracy (fraction of correctly predicted signs) is
    non-differentiable. This loss replaces the hard sign comparison with a
    sigmoid surrogate, making it suitable for gradient-based optimization.

    Notes
    -----
    The surrogate loss is:

    .. math::

        \mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T}
            \sigma(\hat{y}_t \cdot y_t \cdot T_{emp})

    where :math:`\sigma` is the logistic sigmoid, :math:`T_{emp}` is the
    temperature parameter controlling sharpness, and a higher temperature
    produces a harder approximation closer to the true 0/1 accuracy.

    **This is a training proxy** — the value is not comparable to the
    numpy :func:`~fynance.features.stats.directional_accuracy` metric,
    which returns a hard 0/1 fraction.

    Parameters
    ----------
    rf : float, optional
        Not used; inherited from :class:`BaseLoss` for API consistency.
    period : int, optional
        Not used; inherited from :class:`BaseLoss` for API consistency.
    eps : float, optional
        Not used; inherited from :class:`BaseLoss` for API consistency.
    temperature : float, optional
        Sigmoid sharpness. Higher values push the surrogate closer to the
        hard sign indicator. Default is 1.0.

    Examples
    --------
    >>> import torch
    >>> from fynance.models.loss import DirectionalAccuracyLoss
    >>> y_true = torch.tensor([1., -1., 1., -1., 1.])
    >>> y_pred = torch.tensor([0.5, -0.3, 0.1, -0.8, 0.2])
    >>> loss_fn = DirectionalAccuracyLoss()
    >>> loss = loss_fn(y_pred, y_true)
    >>> loss.shape
    torch.Size([])
    >>> loss.item() < 0   # all directions correct → loss close to -1
    True

    See Also
    --------
    SharpeLoss, SortinoLoss

    """

    def __init__(
        self,
        rf: float = 0.,
        period: int = 252,
        eps: float = 1e-8,
        temperature: float = 1.0,
    ):
        super().__init__(rf=rf, period=period, eps=eps)
        self.temperature = temperature

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor,
    ) -> torch.Tensor:
        """ Compute the negative directional accuracy surrogate.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values, shape ``(T,)`` or ``(T, M)``.
        y_true : torch.Tensor
            True values, same shape as ``y_pred``.

        Returns
        -------
        torch.Tensor
            Scalar loss (negative sigmoid-based directional accuracy).

        Raises
        ------
        TypeError
            If ``y_pred`` or ``y_true`` is not a :class:`torch.Tensor`.

        """
        self._check_tensor(y_pred)
        self._check_tensor(y_true)
        return -torch.mean(torch.sigmoid(y_pred * y_true * self.temperature))
