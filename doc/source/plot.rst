-----------------------------------
 Reporting (:mod:`fynance.plot`)
-----------------------------------

Composable matplotlib figures and a one-call :func:`~fynance.plot.tearsheet`
performance report. Each plot returns an ``Axes``/``Figure`` and never calls
``show`` — usable headless, in a notebook, or embedded in an app.

.. currentmodule:: fynance.plot

.. autosummary::
   :toctree: generated/

   tearsheet
   tearsheet_text
   plot_equity
   plot_drawdown
   plot_returns_hist
   plot_rolling_sharpe

Factor tear-sheet
=================

Panels for the factor-evaluation metrics in :mod:`fynance.metrics.factor`, plus
a one-call :func:`~fynance.plot.factor_tearsheet`.

.. autosummary::
   :toctree: generated/

   plot_quantile_returns
   plot_ic_series
   plot_ic_decay
   factor_tearsheet
