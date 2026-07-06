#!/usr/bin/env python3
# coding: utf-8

""" Transaction cost models for the vectorized backtest engine.

Concretizes the :class:`~fynance.core.protocols.CostModel` seam. Four models
ship: :class:`ProportionalCost` (linear, turnover-based), :class:`MarketImpactCost`
(adds a convex, super-linear market-impact term — the square-root impact law),
:class:`HoldingCost` (per-bar carry on the held book: borrow, financing and a
cash credit) and :class:`CompositeCost` (stacks any number of the above, or
any other :class:`~fynance.core.protocols.CostModel`-conforming model, into
one).

"""

from __future__ import annotations

# Built-in packages
from typing import Sequence

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.core.protocols import CostModel
from fynance.portfolio.sizing import transaction_cost

__all__ = ['ProportionalCost', 'MarketImpactCost', 'HoldingCost', 'CompositeCost']


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

    def components(self, weights: NDArray) -> dict[str, NDArray[np.float64]]:
        """ Per-step cost broken down by component (here a single one).

        Returns ``{"transaction": fee+slippage turnover cost}``; the values sum
        to :meth:`__call__`. The optional ``components`` convention lets the
        engine carry a cost breakdown for the tearsheet's cumulative-fees panel.
        """
        return {"transaction": self(weights)}


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

    def components(self, weights: NDArray) -> dict[str, NDArray[np.float64]]:
        """ Per-step cost split into its linear and convex parts.

        Returns ``{"transaction": fee turnover, "market_impact": convex term}``;
        the values sum to :meth:`__call__`. The optional ``components``
        convention lets the engine carry a cost breakdown for the tearsheet's
        cumulative-fees panel.
        """
        turnover = transaction_cost(weights, fee=1.0)

        return {
            "transaction": self.fee * turnover,
            "market_impact": self.impact * turnover ** self.exponent,
        }


class HoldingCost:
    r""" Per-bar carry cost on the held book (not turnover).

    Unlike :class:`ProportionalCost` and :class:`MarketImpactCost`, which
    charge on *trades* (the change between consecutive weights),
    :class:`HoldingCost` charges on the *held* weights at each bar: shorting,
    leverage and idle cash all carry a cost — or, for idle cash, a credit —
    for every bar the position is held, regardless of turnover. Three
    additive terms, each de-annualized to a per-bar rate (``rate / period``):

    - **borrow**: cost of the short gross exposure
      :math:`\sum_i \max(-w_{t,i}, 0)`, at the annualized ``borrow`` rate.
    - **financing**: cost of leverage in excess of fully invested (total
      gross exposure above 1), :math:`\max\big(0, \sum_i |w_{t,i}| - 1\big)`,
      at the annualized ``financing`` rate.
    - **cash**: a *credit* (non-positive) on unused cash,
      :math:`\max\big(0, 1 - \sum_i |w_{t,i}|\big)`, at the annualized
      ``cash_rate``.

    .. math::

        c_t = \frac{borrow}{period} \sum_i \max(-w_{t,i}, 0)
            + \frac{financing}{period}
              \max\Big(0, \sum_i |w_{t,i}| - 1\Big)
            - \frac{cash\_rate}{period}
              \max\Big(0, 1 - \sum_i |w_{t,i}|\Big)

    The total cost at each bar is the sum of the three terms; see
    :meth:`components` for the per-term breakdown. Conforms to
    :class:`~fynance.core.protocols.CostModel`.

    Parameters
    ----------
    borrow : float
        Annualized borrow rate charged on short gross exposure (e.g. ``0.02``
        = 2%/yr). Default 0.0.
    financing : float
        Annualized financing rate charged on leverage above 1x (gross
        exposure in excess of fully invested). Default 0.0.
    cash_rate : float
        Annualized rate credited on unused cash (idle capital not deployed
        into any position). Default 0.0.
    period : int
        Number of bars per year used to de-annualize the rates (e.g. ``252``
        for daily bars, ``12`` for monthly). Default 252.

    Examples
    --------
    >>> import numpy as np
    >>> cost = HoldingCost(borrow=0.02, financing=0.01, cash_rate=0.005, period=250)
    >>> w = np.array([[0.5, 0.0], [-0.5, 1.5], [0.0, 0.0]])
    >>> cost(w)
    array([-1.e-05,  8.e-05, -2.e-05])

    """

    def __init__(
        self,
        borrow: float = 0.0,
        financing: float = 0.0,
        cash_rate: float = 0.0,
        period: int = 252,
    ):
        """ Store the annualized rates and the de-annualization period. """
        self.borrow = borrow
        self.financing = financing
        self.cash_rate = cash_rate
        self.period = period

    def _exposures(self, weights: NDArray) -> tuple[NDArray, NDArray]:
        """ Return per-bar (short gross, total gross) exposure. """
        w = np.asarray(weights, dtype=np.float64)

        if w.ndim == 1:

            return np.maximum(-w, 0.0), np.abs(w)

        return np.maximum(-w, 0.0).sum(axis=1), np.abs(w).sum(axis=1)

    def __call__(self, weights: NDArray) -> NDArray[np.float64]:
        """ Return the per-step (borrow + financing - cash credit) cost. """
        comps = self.components(weights)

        return comps["borrow"] + comps["financing"] + comps["cash"]

    def components(self, weights: NDArray) -> dict[str, NDArray[np.float64]]:
        """ Per-step cost split into its borrow/financing/cash terms.

        Returns ``{"borrow", "financing", "cash"}``; the values sum to
        :meth:`__call__`. ``cash`` entries are non-positive (a credit on idle
        capital). The optional ``components`` convention lets the engine
        carry a cost breakdown for the tearsheet's cumulative-fees panel.
        """
        short, gross = self._exposures(weights)

        return {
            "borrow": (self.borrow / self.period) * short,
            "financing": (self.financing / self.period)
            * np.maximum(0.0, gross - 1.0),
            "cash": -(self.cash_rate / self.period)
            * np.maximum(0.0, 1.0 - gross),
        }


class CompositeCost:
    """ Stack several cost models into one.

    Sums the per-step cost of every model in ``models`` (each of which must
    conform to :class:`~fynance.core.protocols.CostModel`), so a single
    :class:`CompositeCost` instance can be passed to
    :func:`~fynance.backtest.engine.backtest` wherever one ``cost`` model is
    expected. Conforms to :class:`~fynance.core.protocols.CostModel` itself.

    Parameters
    ----------
    models : sequence of CostModel
        The cost models to stack, in the order their components are merged
        by :meth:`components`.

    Examples
    --------
    >>> import numpy as np
    >>> cost = CompositeCost(
    ...     [ProportionalCost(fee=0.001), HoldingCost(borrow=0.02, period=250)]
    ... )
    >>> w = np.array([[1.0, 0.0], [-0.5, 0.5], [-0.5, 0.5]])
    >>> cost(w)
    array([1.00e-03, 2.04e-03, 4.00e-05])

    """

    def __init__(self, models: Sequence[CostModel]):
        """ Store the sequence of cost models to stack. """
        self.models = list(models)

    def __call__(self, weights: NDArray) -> NDArray[np.float64]:
        """ Return the per-step cost summed over every stacked model. """
        w = np.asarray(weights, dtype=np.float64)
        n = w.shape[0]
        total = np.zeros(n, dtype=np.float64)

        for model in self.models:
            total = total + np.asarray(model(weights), dtype=np.float64)

        return total

    def components(self, weights: NDArray) -> dict[str, NDArray[np.float64]]:
        """ Merge the per-step components of every stacked model.

        Each model contributes its own :meth:`components` breakdown (or, if a
        model does not expose ``components``, its total under a key equal to
        its class name). Component keys are merged in ``models`` order; when a
        key already produced by an earlier model recurs, the *later*
        occurrence is disambiguated by prefixing it with its owning class
        name, e.g. ``"transaction"`` then ``"MarketImpactCost.transaction"``.
        A third collision on the same class+key (e.g. several stacked
        ``ProportionalCost``) is further disambiguated by the model's index,
        ``"ProportionalCost[2].transaction"``, so no contribution is ever
        overwritten. The merged values always sum to :meth:`__call__`.
        """
        merged: dict[str, NDArray[np.float64]] = {}

        for i, model in enumerate(self.models):
            cls_name = type(model).__name__
            if hasattr(model, "components"):
                comps = model.components(weights)

            else:
                comps = {cls_name: model(weights)}

            for key, value in comps.items():
                out_key = key
                if out_key in merged:
                    out_key = f"{cls_name}.{key}"
                if out_key in merged:
                    out_key = f"{cls_name}[{i}].{key}"

                merged[out_key] = np.asarray(value, dtype=np.float64)

        return merged
