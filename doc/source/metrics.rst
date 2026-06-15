-------------------------------------
 Performance metrics (:mod:`fynance.metrics`)
-------------------------------------

Risk-adjusted ratios, return and drawdown metrics to evaluate a strategy
out-of-sample. A metric *evaluates* a series — distinct from
:mod:`fynance.features`, which *builds* inputs.

.. currentmodule:: fynance.metrics

Ratios & returns
================

.. autosummary::
   :toctree: generated/

   sharpe
   sortino
   calmar
   annual_return
   annual_volatility
   diversified_ratio
   perf_index
   perf_returns
   perf_strat
   returns_strat

Drawdown
========

.. autosummary::
   :toctree: generated/

   drawdown
   mdd

Rolling versions
================

.. autosummary::
   :toctree: generated/

   roll_sharpe
   roll_calmar
   roll_annual_return
   roll_annual_volatility
   roll_drawdown
   roll_mdd

Aggregated report
=================

.. autosummary::
   :toctree: generated/

   summary
