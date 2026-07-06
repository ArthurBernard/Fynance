*********************
Rebalancing & delay
*********************

Composable, strictly causal ``(T, N)`` transforms between an allocator/signal and :func:`~fynance.backtest.engine.backtest`: the effective book drifts with asset returns and trading is throttled on a calendar (:func:`~fynance.portfolio.rebalance.rebalance_calendar`), inside a no-trade band (:func:`~fynance.portfolio.rebalance.rebalance_band`) or under a per-bar turnover budget (:func:`~fynance.portfolio.rebalance.rebalance_turnover_cap`), plus whole-lot discretization (:func:`~fynance.portfolio.rebalance.discretize`) and an execution delay (:func:`~fynance.portfolio.rebalance.delay`).

.. currentmodule:: fynance.portfolio.rebalance

.. autosummary::
   :toctree: generated/

   rebalance_calendar
   rebalance_band
   rebalance_turnover_cap
   discretize
   delay
