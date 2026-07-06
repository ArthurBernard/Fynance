***************
Position sizing
***************

Position sizing and transaction-cost primitives for realistic backtests: fractional Kelly (:func:`~fynance.portfolio.sizing.kelly_fraction`), volatility targeting for a single series (:func:`~fynance.portfolio.sizing.vol_target`) or a whole ``(T, N)`` book (:func:`~fynance.portfolio.sizing.book_vol_target`) and turnover-based transaction costs (:func:`~fynance.portfolio.sizing.transaction_cost`).

.. currentmodule:: fynance.portfolio.sizing

.. autosummary::
   :toctree: generated/

   kelly_fraction
   vol_target
   book_vol_target
   transaction_cost
