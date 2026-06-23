#!/usr/bin/env python3
# coding: utf-8

""" Base class for financial loss functions.

Defines :class:`BaseLoss`, the common foundation for all differentiable
loss functions in :mod:`fynance.models.loss`.

"""

from __future__ import annotations

# Third-party packages
import torch
import torch.nn

__all__ = ['BaseLoss']

#: Hard magnitude bound for ratio-style losses (Sharpe / Sortino / Calmar /
#: Omega). It is far above any plausible real ratio (which are O(1)-O(10)), so
#: it never touches a normal-case value; it only caps the runaway that a
#: collapsing risk denominator would otherwise produce, keeping the loss finite
#: and its gradients well-scaled.
MAX_RATIO: float = 1e3


class BaseLoss(torch.nn.Module):
    """ Base class for differentiable financial loss functions.

    Holds the shared hyper-parameters (risk-free rate, annualization period,
    numerical stabilizer) and enforces that inputs are :class:`torch.Tensor`.
    Subclass and implement :meth:`forward` to define a new loss.

    Parameters
    ----------
    rf : float, optional
        Annualized risk-free rate. Default is 0.
    period : int, optional
        Number of periods per year used for annualization. Default is 252.
    eps : float, optional
        Small constant added to denominators to avoid division by zero.
        Default is 1e-8.

    """

    def __init__(self, rf: float = 0., period: int = 252, eps: float = 1e-8):
        super().__init__()
        self.rf = rf
        self.period = period
        self.eps = eps

    @property
    def _rf_per_period(self) -> float:
        """ Per-period risk-free rate, recomputed from the live ``rf``/``period``.

        Computed as a property (not cached in ``__init__``) so that mutating
        ``loss.rf`` or ``loss.period`` after construction takes effect on the
        next :meth:`forward` instead of being a silent no-op.
        """
        return self.rf / self.period

    def _check_tensor(self, x: object) -> None:
        """ Raise TypeError if *x* is not a :class:`torch.Tensor`. """
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, got {type(x).__name__}. "
                "Convert numpy arrays with torch.from_numpy() before passing "
                "to this loss."
            )

    @staticmethod
    def _book_return(y_pred: torch.Tensor) -> torch.Tensor:
        r""" Aggregate a position-book return ``(T, N)`` into a 1-D book return.

        A 2-D ``y_pred`` is read as the per-asset net-of-cost returns of a
        position book; the book return at each step is their sum across assets
        (:math:`\sum_i pos_i \cdot r_i`), so the ratio losses score the
        **portfolio** return rather than a pooled per-asset return. A 1-D
        ``y_pred`` (single asset) is returned unchanged, and a ``(T, 1)`` book
        reduces to the same series as the single-asset case.
        """
        if y_pred.dim() == 2:

            return y_pred.sum(dim=1)

        return y_pred
