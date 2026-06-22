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
    The downside is therefore floored with a **returns-scaled** value
    ``|excess|.mean() / MAX_RATIO`` (plus a bare ``eps`` backstop for the
    degenerate all-zero batch). This caps the ratio at roughly
    ``MAX_RATIO`` in the low-downside regime regardless of the return
    scale, keeping the loss finite and bounded while preserving the sign
    convention.

    The ratio is then passed through a **smooth saturating map**,
    ``MAX_RATIO * tanh(ratio / MAX_RATIO)``, instead of a hard clamp. A
    hard clamp pins the loss to a constant on a low-downside batch and so
    kills the gradient in exactly the strong-uptrend regime training still
    wants to push on; ``tanh`` is near-linear for normal-regime ratios
    (leaving their numerics unchanged) yet keeps a residual, non-zero
    gradient when the ratio is large.

    Parameters
    ----------
    rf : float, optional
        Annualized risk-free rate. Default is 0.
    period : int, optional
        Number of periods per year. Default is 252.
    eps : float, optional
        Bare numerical stabilizer added to the returns-scaled downside floor
        as a backstop for the degenerate all-zero batch (see Notes).
        Default is 1e-8.

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
        # lets the ratio explode on an all-gains batch. Scaling the floor by
        # ``|excess|.mean() / MAX_RATIO`` caps the ratio at ~MAX_RATIO in the
        # low-downside regime (scale-invariantly); ``eps`` backstops the
        # degenerate all-zero case.
        floor = excess.abs().mean() / _MAX_RATIO + self.eps
        ratio = excess.mean() / torch.clamp(downside, min=floor)
        # Smooth saturating map instead of a hard clamp: tanh is near-linear for
        # normal-regime ratios (numerics unchanged) but keeps a non-zero
        # gradient when the ratio is large, unlike a clamp that zeroes it.
        return -_MAX_RATIO * torch.tanh(ratio / _MAX_RATIO)
