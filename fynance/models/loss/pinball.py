#!/usr/bin/env python3
# coding: utf-8

""" Differentiable pinball (quantile) loss for PyTorch training loops. """

from __future__ import annotations

# Built-in packages
from collections.abc import Sequence

# Third-party packages
import torch

# Local packages
from ._base import BaseLoss

__all__ = ['PinballLoss']


def _validate_taus(taus: float | Sequence[float]) -> tuple[float, ...]:
    """ Validate ``taus`` and return them as a sorted tuple of floats in (0, 1).

    Accepts a single float or any sequence of floats; a single float is
    wrapped into a one-element tuple. The returned tuple is sorted
    ascending so every caller (:class:`PinballLoss`,
    :class:`~fynance.models.quantile.QuantileModel`) shares one canonical
    column order for the quantile axis, regardless of the order ``taus``
    was given in.
    """
    if isinstance(taus, (int, float)):
        taus = (float(taus),)

    values = tuple(float(t) for t in taus)

    if not values:
        raise ValueError("taus must be non-empty")

    if any(not (0. < t < 1.) for t in values):
        raise ValueError(f"each tau must be strictly in (0, 1), got {values}")

    return tuple(sorted(values))


class PinballLoss(BaseLoss):
    r""" Pinball (quantile / check) loss for multi-quantile regression.

    For a target quantile :math:`\tau \in (0, 1)` and error
    :math:`e = y - \hat{y}_\tau`:

    .. math::

        \mathcal{L}_\tau(e) = \max(\tau e,\ (\tau - 1) e)

    an asymmetric penalty: under-prediction (:math:`e > 0`) is penalized by
    :math:`\tau` and over-prediction (:math:`e < 0`) by :math:`1 - \tau`. At
    :math:`\tau = 0.5` it reduces to :math:`0.5 |e|`, i.e. half the mean
    absolute error. Minimizing it over a batch drives ``pred`` toward the
    :math:`\tau`-th conditional quantile of ``target``.

    With several ``taus`` at once, ``pred`` carries one column per quantile
    and the reported loss is the mean over quantiles *and* samples of the
    per-quantile pinball loss — the standard way to train a single
    multi-quantile head such as
    :class:`~fynance.models.quantile.QuantileModel`.

    Notes
    -----
    ``taus`` are sorted ascending internally (see :func:`_validate_taus`),
    so ``pred``'s trailing axis is expected in that same ascending order.
    This loss does **not** enforce non-crossing quantiles (``q10 <= q50 <=
    q90``) — it is the plain per-column pinball loss; monotonicity is a
    predict-time concern (see
    :meth:`~fynance.models.quantile.QuantileModel.predict_quantiles`).

    Parameters
    ----------
    taus : float or sequence of float, optional
        Target quantile(s) in ``(0, 1)``. Default ``(0.1, 0.5, 0.9)``.
    **kwargs
        Forwarded to :class:`BaseLoss` (``rf``, ``period``, ``eps``) — not
        used by this loss (it is not a ratio/annualized objective), kept
        only for API consistency across :mod:`fynance.models.loss`.

    Raises
    ------
    ValueError
        If ``taus`` is empty or any value is not strictly in ``(0, 1)``.

    Examples
    --------
    >>> import torch
    >>> from fynance.models.loss import PinballLoss
    >>> target = torch.tensor([1.0, 2.0, 3.0])
    >>> pred = torch.tensor([[1.0], [2.0], [3.0]])   # perfect single-tau fit
    >>> PinballLoss(taus=0.5)(pred, target).item()
    0.0
    >>> pred_off = pred + 1.0
    >>> loss = PinballLoss(taus=0.5)(pred_off, target)
    >>> loss.item() == 0.5   # tau=0.5 -> half the (constant) absolute error
    True

    See Also
    --------
    fynance.models.quantile.QuantileModel

    """

    def __init__(
        self,
        taus: float | Sequence[float] = (0.1, 0.5, 0.9),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.taus = _validate_taus(taus)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        r""" Compute the mean pinball loss across quantiles and samples.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted quantiles, shape ``(..., n_taus)``.
        target : torch.Tensor
            True values, shape ``(...,)`` — ``pred`` without its trailing
            quantile axis — broadcast against every quantile column.

        Returns
        -------
        torch.Tensor
            Scalar loss: the mean over quantiles and samples of
            :math:`\max(\tau e, (\tau - 1) e)`.

        Raises
        ------
        TypeError
            If ``pred`` or ``target`` is not a :class:`torch.Tensor`.
        ValueError
            If ``pred``'s trailing dimension does not equal the number of
            ``taus``, or ``target``'s shape does not match ``pred``'s
            shape without that trailing dimension.

        """
        self._check_tensor(pred)
        self._check_tensor(target)
        n_taus = len(self.taus)

        if pred.shape[-1] != n_taus:
            raise ValueError(
                f"pred's trailing dimension ({pred.shape[-1]}) must equal "
                f"the number of taus ({n_taus})"
            )

        if tuple(target.shape) != tuple(pred.shape[:-1]):
            raise ValueError(
                f"target shape {tuple(target.shape)} must equal pred's "
                f"shape without the trailing quantile axis "
                f"{tuple(pred.shape[:-1])}"
            )

        taus = torch.as_tensor(self.taus, dtype=pred.dtype, device=pred.device)
        e = target.unsqueeze(-1) - pred

        return torch.maximum(taus * e, (taus - 1.) * e).mean()
