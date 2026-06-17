*************
Market regime
*************

Unsupervised market-regime labelling by clustering rolling volatility / return
features (built by :func:`~fynance.features.regime.regime_features`).
:func:`~fynance.features.regime.detect_regimes` is the in-sample convenience (for
analysis only — it sees the whole series).

For backtests use :class:`~fynance.features.regime.RegimeDetector`, the **causal**
variant: ``fit`` clusters on the **train** window only, then ``predict`` assigns
each later bar to its nearest centroid **online** — no future information leaks
into a label. Use the resulting regime as a feature column in the ``X`` matrix you
pass to a strategy (see :doc:`strategy` and :doc:`research_workflow`).

.. currentmodule:: fynance.features.regime

.. autosummary::
   :toctree: generated/

   detect_regimes
   regime_features
   RegimeDetector
