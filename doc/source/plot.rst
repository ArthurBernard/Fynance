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
