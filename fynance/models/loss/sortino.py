#!/usr/bin/env python3
# coding: utf-8

""" Differentiable Sortino ratio loss for PyTorch training loops. """

from __future__ import annotations

# Third-party packages
import torch
import torch.nn.functional as F

# Local packages
from ._base import MAX_RATIO as _MAX_RATIO
from ._base import BaseLoss

__all__ = ['SortinoLoss']


class SortinoLoss(BaseLoss):
    r""" Negative Sortino ratio as a differentiable loss.

    Minimizing this loss penalizes downside returns only, unlike
    :class:`SharpeLoss` which penalizes both tails symmetrically.

    Notes
    -----
    The loss is defined as:

    .. math::

        \mathcal{L} = -\frac{\mu(r - rf_p)}
            {\sqrt{\mu(\text{ReLU}(-(r - rf_p))^2) + \varepsilon}}

    where :math:`r` is ``y_pred`` and :math:`rf_p = rf / period`.
    The denominator is a differentiable proxy for the downside deviation;
    its magnitude differs from the numpy
    :func:`~fynance.metrics.sortino` evaluation metric.

    **This is a training proxy** — the value is not comparable to the
    numpy :func:`~fynance.metrics.sortino` evaluation metric.

    The downside deviation is :math:`O(\text{returns})`, so a fixed
    absolute ``eps`` inside the square root is dimensionally wrong: on an
    all-gains batch (zero downside) the loss would explode (e.g. ``-100``).
    The downside is therefore floored with a **returns-scaled** epsilon
    proportional to the excess-return magnitude, keeping the loss finite
    and bounded while preserving the sign convention.

    Parameters
    ----------
    rf : float, optional
        Annualized risk-free rate. Default is 0.
    period : int, optional
        Number of periods per year. Default is 252.
    eps : float, optional
        Relative numerical stabilizer scaling the downside-deviation floor
        (see Notes). Default is 1e-8.

    Examples
    --------
    >>> import torch
    >>> from fynance.models.loss import SortinoLoss
    >>> returns = torch.tensor([-0.01, 0.02, 0.01, -0.005, 0.03])
    >>> loss_fn = SortinoLoss()
    >>> loss = loss_fn(returns)
    >>> loss.shape
    torch.Size([])

    See Also
    --------
    SharpeLoss, DirectionalAccuracyLoss

    """

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Compute the negative Sortino ratio.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted return series, shape ``(T,)`` or ``(T, M)``.
        y_true : torch.Tensor, optional
            Not used; accepted for API compatibility with PyTorch criterions.

        Returns
        -------
        torch.Tensor
            Scalar loss value (negative Sortino ratio proxy).

        Raises
        ------
        TypeError
            If ``y_pred`` is not a :class:`torch.Tensor`.

        """
        self._check_tensor(y_pred)
        excess = y_pred - self._rf_per_period
        downside = torch.sqrt(torch.mean(F.relu(-excess) ** 2))
        # Floor the downside relative to the return scale: a fixed absolute eps
        # inside the sqrt is dimensionally wrong for an O(returns) downside and
        # lets the ratio explode on an all-gains batch. ``eps`` backstops the
        # degenerate all-zero case; the final clamp bounds the magnitude in the
        # near-zero-downside regime (where pushing for more gains is fine, so
        # saturating the gradient there is harmless for this training proxy).
        floor = self.eps * excess.abs().mean() + self.eps
        ratio = excess.mean() / torch.clamp(downside, min=floor)
        return -torch.clamp(ratio, min=-_MAX_RATIO, max=_MAX_RATIO)
