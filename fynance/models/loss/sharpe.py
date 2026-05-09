#!/usr/bin/env python3
# coding: utf-8

""" Differentiable Sharpe ratio loss for PyTorch training loops. """

from __future__ import annotations

# Third-party packages
import torch

# Local packages
from ._base import BaseLoss

__all__ = ['SharpeLoss']


class SharpeLoss(BaseLoss):
    r""" Negative Sharpe ratio as a differentiable loss.

    Minimizing this loss is equivalent to maximizing the Sharpe ratio of
    the predicted return series. All operations are pure PyTorch, so
    gradients flow through ``forward`` without any NumPy conversion.

    Notes
    -----
    The loss is defined as:

    .. math::

        \mathcal{L} = -\frac{\mu(r - rf_p)}{\sigma(r - rf_p) + \varepsilon}

    where :math:`r` is ``y_pred``, :math:`rf_p = rf / period` is the
    per-period risk-free rate, :math:`\mu` and :math:`\sigma` are the
    mean and population standard deviation (``correction=0``, consistent
    with :func:`~fynance.features.metrics.sharpe`), and :math:`\varepsilon`
    is the numerical stabilizer (``eps``).

    **This is a training proxy** — the value is not comparable to the
    numpy :func:`~fynance.features.metrics.sharpe` evaluation metric,
    which annualizes over a price series.

    Parameters
    ----------
    rf : float, optional
        Annualized risk-free rate. Default is 0.
    period : int, optional
        Number of periods per year. Default is 252.
    eps : float, optional
        Numerical stabilizer. Default is 1e-8.

    Examples
    --------
    >>> import torch
    >>> from fynance.models.loss import SharpeLoss
    >>> torch.manual_seed(0)
    <...>
    >>> returns = torch.randn(50) * 0.01 + 0.001
    >>> loss_fn = SharpeLoss()
    >>> loss = loss_fn(returns)
    >>> loss.shape
    torch.Size([])
    >>> loss.item() < 0       # positive mean excess return → negative loss
    True

    Using with :class:`~fynance.models.mlp.MultiLayerPerceptron`:

    >>> from fynance.models import MultiLayerPerceptron, SharpeLoss
    >>> import torch.optim
    >>> X = torch.randn(100, 4)
    >>> y = torch.randn(100, 1)
    >>> model = MultiLayerPerceptron(X, y, layers=[16])
    >>> _ = model.set_optimizer(SharpeLoss, torch.optim.Adam, lr=1e-3)
    >>> loss = model.train_on(X, y)
    >>> loss.shape
    torch.Size([])

    See Also
    --------
    SortinoLoss, DirectionalAccuracyLoss

    """

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Compute the negative Sharpe ratio.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted return series, shape ``(T,)`` or ``(T, M)``.
        y_true : torch.Tensor, optional
            Not used; accepted for API compatibility with PyTorch criterions.

        Returns
        -------
        torch.Tensor
            Scalar loss value (negative Sharpe ratio).

        Raises
        ------
        TypeError
            If ``y_pred`` is not a :class:`torch.Tensor`.

        """
        self._check_tensor(y_pred)
        excess = y_pred - self._rf_per_period
        return -(excess.mean() / (excess.std(correction=0) + self.eps))
