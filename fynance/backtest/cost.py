#!/usr/bin/env python3
# coding: utf-8

""" Transaction cost models for the vectorized backtest engine.

Concretizes the :class:`~fynance.core.protocols.CostModel` seam. Only the
proportional (turnover-based) model ships in 2.0; non-linear slippage is a
documented extension point.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.algorithms.sizing import transaction_cost

__all__ = ['ProportionalCost']


class ProportionalCost:
    """ Proportional transaction cost: ``(fee + slippage) * turnover``.

    Turnover at each step is the absolute traded weight
    :math:`\\sum_i |w_{t,i} - w_{t-1,i}|` (the first step charges the initial
    position). Conforms to :class:`~fynance.core.protocols.CostModel`.

    Parameters
    ----------
    fee : float
        Proportional fee per unit traded (e.g. ``0.001`` = 10 bps).
    slippage : float
        Additional proportional slippage per unit traded.

    Examples
    --------
    >>> import numpy as np
    >>> cost = ProportionalCost(fee=0.01)
    >>> cost(np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]]))
    array([0.01, 0.01, 0.  ])

    """

    def __init__(self, fee: float = 0.0, slippage: float = 0.0):
        """ Store the cost rates. """
        self.fee = fee
        self.slippage = slippage

    def __call__(self, weights: NDArray) -> NDArray[np.float64]:
        """ Return the per-step proportional cost of a weight book. """
        rate = self.fee + self.slippage

        if rate == 0.0:
            w = np.asarray(weights, dtype=np.float64)

            return np.zeros(w.shape[0], dtype=np.float64)

        return transaction_cost(weights, fee=rate)
