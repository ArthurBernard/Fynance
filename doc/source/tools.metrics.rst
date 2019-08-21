**************
Metrics module
**************
Some tools to compute financial metrics.

.. currentmodule:: fynance.tools.metrics

.. autosummary::

   fynance.tools.metrics.accuracy
   fynance.tools.metrics.calmar
   fynance.tools.metrics.diversified_ratio
   fynance.tools.metrics.drawdown
   fynance.tools.metrics.mad
   fynance.tools.metrics.mdd
   fynance.tools.metrics.perf_index
   fynance.tools.metrics.perf_returns
   fynance.tools.metrics.roll_calmar
   fynance.tools.metrics.roll_mad
   fynance.tools.metrics.roll_sharpe
   fynance.tools.metrics.roll_z_score
   fynance.tools.metrics.sharpe
   fynance.tools.metrics.z_score

Scalar functions
================

The following functions return a scalar.

.. autofunction:: accuracy

.. autofunction:: calmar

.. autofunction:: diversified_ratio

.. autofunction:: mad

.. autofunction:: mdd

.. autofunction:: sharpe

.. autofunction:: z_score

Vectorized functions
====================

The following functions return a vector.

.. autofunction:: drawdown

.. autofunction:: perf_index

.. autofunction:: perf_returns

.. autofunction:: roll_calmar

.. autofunction:: roll_mad

.. autofunction:: roll_sharpe

.. autofunction:: roll_z_score