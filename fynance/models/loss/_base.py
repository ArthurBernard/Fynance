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

    def _check_tensor(self, x: object) -> None:
        """ Raise TypeError if *x* is not a :class:`torch.Tensor`. """
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, got {type(x).__name__}. "
                "Convert numpy arrays with torch.from_numpy() before passing "
                "to this loss."
            )
