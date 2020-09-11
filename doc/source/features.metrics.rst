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

   fynance.features.metrics.accuracy
   fynance.features.metrics.annual_return
   fynance.features.metrics.annual_volatility
   fynance.features.metrics.calmar
   fynance.features.metrics.diversified_ratio
   fynance.features.metrics.drawdown
   fynance.features.metrics.mad
   fynance.features.metrics.mdd
   fynance.features.metrics.perf_index
   fynance.features.metrics.perf_returns
   fynance.features.metrics.perf_strat
   fynance.features.metrics.sharpe
   fynance.features.metrics.z_score

Rolling version of metrics
==========================

.. autosummary::
   :toctree: generated/

   fynance.features.metrics.roll_annual_return
   fynance.features.metrics.roll_annual_volatility
   fynance.features.metrics.roll_calmar
   fynance.features.metrics.roll_drawdown
   fynance.features.metrics.roll_mad
   fynance.features.metrics.roll_mdd
   fynance.features.metrics.roll_sharpe
   fynance.features.metrics.roll_z_score
