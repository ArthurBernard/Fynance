*********
 Metrics 
*********

This module contains some tools to compute financial metrics, such as annualized returns (:func:`~fynance.features.metrics.annual_return`), annualized volatility (:func:`~fynance.features.metrics.annual_volatility`), Calmar ratio (:func:`~fynance.features.metrics.calmar`), diversification ratio (:func:`~fynance.features.metrics.diversified_ratio`), maximum drawdown (:func:`~fynance.features.metrics.mdd`), Sharpe ratio (:func:`~fynance.features.metrics.sharpe`), Z-score (:func:`~fynance.features.metrics.z_score`), etc.

There is also rolling version of some metrics, annualized returns (:func:`~fynance.features.metrics.roll_annual_return`), annualized volatility (:func:`~fynance.features.metrics.roll_annual_volatility`), Calmar ratio (:func:`~fynance.features.metrics.roll_calmar`), maximum drawdown (:func:`~fynance.features.metrics.roll_mdd`), Sharpe ratio (:func:`~fynance.features.metrics.roll_sharpe`), Z-score (:func:`~fynance.features.metrics.roll_z_score`), etc.

.. currentmodule:: fynance.features.metrics

Classical version of metrics
============================

.. autosummary::
   :toctree: generated/

   accuracy
   annual_return
   annual_volatility
   calmar
   diversified_ratio
   drawdown
   mad
   mdd
   percent_positive
   perf_index
   perf_returns
   perf_strat
   sharpe
   tail_ratio
   z_score

Rolling version of metrics
==========================

.. autosummary::
   :toctree: generated/

   roll_annual_return
   roll_annual_volatility
   roll_calmar
   roll_drawdown
   roll_mad
   roll_mdd
   roll_sharpe
   roll_z_score
