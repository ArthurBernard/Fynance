#!/usr/bin/env python3
# coding: utf-8

""" Transaction cost models for the vectorized backtest engine.

Concretizes the :class:`~fynance.core.protocols.CostModel` seam. Two models
ship: :class:`ProportionalCost` (linear, turnover-based) and
:class:`MarketImpactCost` (adds a convex, super-linear market-impact term — the
square-root impact law).

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.portfolio.sizing import transaction_cost

__all__ = ['ProportionalCost', 'MarketImpactCost']


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


class MarketImpactCost:
    """ Non-linear market-impact cost: linear fee + convex impact term.

    Charges, per step, a proportional fee plus a **super-linear** market-impact
    term on the same turnover (the standard square-root impact law, where the
    cost grows faster than the trade size):

    .. math::

        c_t = fee \\cdot \\tau_t + impact \\cdot \\tau_t^{\\,exponent}

    where the turnover :math:`\\tau_t = \\sum_i |w_{t,i} - w_{t-1,i}|` is defined
    exactly as in :class:`ProportionalCost` (the first step charges the initial
    position). With ``exponent=1`` and ``impact=0`` it reduces to
    :class:`ProportionalCost`. Conforms to
    :class:`~fynance.core.protocols.CostModel`.

    Parameters
    ----------
    fee : float
        Proportional (linear) fee per unit traded (e.g. ``0.001`` = 10 bps).
    impact : float
        Coefficient of the convex impact term, in the same units as ``fee``.
    exponent : float
        Convexity exponent of the impact term (``> 1`` is super-linear).
        Default ``1.5`` (the square-root impact law: cost per unit traded grows
        as :math:`\\sqrt{\\tau}`).

    Examples
    --------
    >>> import numpy as np
    >>> cost = MarketImpactCost(fee=0.0, impact=0.1, exponent=2.0)
    >>> cost(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    array([0.1, 0.4, 0. ])

    """

    def __init__(
        self,
        fee: float = 0.0,
        impact: float = 0.0,
        exponent: float = 1.5,
    ):
        """ Store the cost rates and the impact convexity exponent. """
        if exponent <= 0:

            raise ValueError(f"exponent must be positive, got {exponent}")

        self.fee = fee
        self.impact = impact
        self.exponent = exponent

    def __call__(self, weights: NDArray) -> NDArray[np.float64]:
        """ Return the per-step (linear + convex impact) cost of a weight book. """
        # transaction_cost with unit fee returns the raw per-step turnover.
        turnover = transaction_cost(weights, fee=1.0)

        return self.fee * turnover + self.impact * turnover ** self.exponent
