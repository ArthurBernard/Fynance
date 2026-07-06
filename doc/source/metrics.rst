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

Tail risk
=========

Value-at-Risk, Conditional Value-at-Risk (Expected Shortfall) and Conditional
Drawdown-at-Risk. Like the ratios above, ``var``/``cvar``/``cdar`` take a
price/equity curve (returns are derived internally); ``tail_dependence`` is the
exception — it takes a ``(T, N)`` returns panel directly, mirroring
:func:`information_coefficient`'s pair convention.

.. autosummary::
   :toctree: generated/

   var
   cvar
   cdar
   tail_dependence

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
   roll_var
   roll_cvar

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

Trading exposure & turnover
============================

Position-level analytics — how a book trades (churn) and sits (leverage,
long/short bias) — taking a weight/position series rather than an equity
curve, so (like the factor helpers above) these are intentionally kept out
of the ``METRICS`` registry.

.. autosummary::
   :toctree: generated/

   turnover_series
   annual_turnover
   gross_exposure
   net_exposure
   exposure_summary

Aggregated report
=================

.. autosummary::
   :toctree: generated/

   summary

:func:`summary` is driven by the ``METRICS`` registry — a name → callable
mapping (``annual_return``, ``annual_volatility``, ``sharpe``, ``sortino``,
``calmar``, ``max_drawdown``, ``var``, ``cvar``, ``cdar``) you can read or
extend.

.. autodata:: METRICS
   :no-value:
