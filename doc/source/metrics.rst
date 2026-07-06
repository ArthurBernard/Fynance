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

Benchmark-relative
==================

Score a strategy *against* a benchmark rather than in isolation: beta and
Jensen's alpha decompose the strategy's return into a benchmark-driven part
and a residual, while tracking error, the information ratio and the up/down
capture ratios describe the active (strategy-minus-benchmark) return. Every
function takes two aligned price/level curves ``(X, B)`` (see
:mod:`fynance.metrics.benchmark`).

.. autosummary::
   :toctree: generated/

   beta
   alpha
   tracking_error
   information_ratio
   capture_ratio
   benchmark_summary
   roll_beta_benchmark

Factor analysis
===============

Alphalens-style evaluation of a cross-sectional factor on a data-agnostic
``(T, N)`` panel. The alignment convention matches
:func:`information_coefficient` — ``factor[t]`` is paired with the return
realized *after* the factor is known.

.. autosummary::
   :toctree: generated/

   information_coefficient
   quantile_returns
   roll_information_coefficient
   ic_decay
   ic_summary
   factor_rank_autocorr
   QuantileResult

Aggregated report
=================

.. autosummary::
   :toctree: generated/

   summary

:func:`summary` is driven by the ``METRICS`` registry — a name → callable
mapping (``annual_return``, ``annual_volatility``, ``sharpe``, ``sortino``,
``calmar``, ``max_drawdown``) you can read or extend.

.. autodata:: METRICS
   :no-value:
