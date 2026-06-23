#!/usr/bin/env python3
# coding: utf-8

""" Differentiable cross-sectional ranking loss. """

from __future__ import annotations

# Third-party packages
import torch

# Local packages
from ._base import BaseLoss

__all__ = ['RankingLoss']


class RankingLoss(BaseLoss):
    r""" Differentiable cross-sectional long-short ranking loss.

    A *predict-then-rank* objective for a panel of assets: it rewards a score
    that ranks assets by their realized cross-sectional outcome. At each time
    step the per-asset scores are turned into a softmax **long** book and a
    softmax **short** book (a smooth, differentiable relaxation of "long the
    top names, short the bottom names"); the loss is the negative mean
    long-short **spread return**

    .. math::

        \mathcal{L} = -\frac{1}{T} \sum_t \sum_i
            \big(w^{+}_{t,i} - w^{-}_{t,i}\big)\, r_{t,i},
        \qquad
        w^{\pm}_{t} = \mathrm{softmax}(\pm\,\tau\, s_t)

    where :math:`s` is ``y_pred`` (the per-asset scores), :math:`r` is
    ``y_true`` (the realized per-asset returns) and :math:`\tau` is
    ``temperature`` (higher = sharper, closer to a hard top/bottom selection).
    Minimizing the loss maximizes the spread: it pushes high scores onto the
    cross-sectional winners and low scores onto the losers.

    Unlike the ratio losses, this one is **inherently cross-sectional**: it
    needs a 2-D ``(T, N)`` panel (``N >= 2`` assets) and a realized target.

    Parameters
    ----------
    temperature : float, optional
        Softmax sharpness :math:`\tau`. Higher pushes the soft long/short books
        toward a hard top/bottom selection. Default is 1.0.
    **kwargs
        Forwarded to :class:`BaseLoss` (``rf``, ``period``, ``eps``).

    Examples
    --------
    >>> import torch
    >>> from fynance.models.loss import RankingLoss
    >>> scores = torch.tensor([[3., 2., 1.], [1., 2., 3.]])
    >>> real = torch.tensor([[0.03, 0.0, -0.03], [-0.03, 0.0, 0.03]])
    >>> loss_fn = RankingLoss()
    >>> aligned = loss_fn(scores, real)
    >>> inverted = loss_fn(-scores, real)
    >>> bool(aligned < inverted)        # scoring the winners high ranks best
    True

    See Also
    --------
    SharpeLoss, SortinoLoss

    """

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor,
    ) -> torch.Tensor:
        """ Compute the negative mean cross-sectional long-short spread.

        Parameters
        ----------
        y_pred : torch.Tensor
            Per-asset scores, shape ``(T, N)`` with ``N >= 2``.
        y_true : torch.Tensor
            Realized per-asset returns, same shape as ``y_pred``.

        Returns
        -------
        torch.Tensor
            Scalar loss (negative mean long-short spread return).

        Raises
        ------
        TypeError
            If ``y_pred`` or ``y_true`` is not a :class:`torch.Tensor`.
        ValueError
            If the inputs are not matching-shape 2-D panels with ``N >= 2``.

        """
        self._check_tensor(y_pred)
        self._check_tensor(y_true)

        if y_pred.dim() != 2 or y_pred.shape[1] < 2:

            raise ValueError(
                "RankingLoss needs a 2-D (T, N) panel with N >= 2 assets, got "
                f"shape {tuple(y_pred.shape)}"
            )

        if y_pred.shape != y_true.shape:

            raise ValueError(
                "y_pred and y_true must have the same shape, got "
                f"{tuple(y_pred.shape)} and {tuple(y_true.shape)}"
            )

        w_long = torch.softmax(self.temperature * y_pred, dim=1)
        w_short = torch.softmax(-self.temperature * y_pred, dim=1)
        spread = ((w_long - w_short) * y_true).sum(dim=1)

        return -spread.mean()
