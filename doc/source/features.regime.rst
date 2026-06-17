*************
Market regime
*************

Unsupervised market-regime labelling by clustering rolling volatility / return
features. :func:`~fynance.features.regime.detect_regimes` is the in-sample
convenience (for analysis); :class:`~fynance.features.regime.RegimeDetector` is
the **causal** fit-on-train / assign-online variant for backtests.

.. currentmodule:: fynance.features.regime

.. autosummary::
   :toctree: generated/

   detect_regimes
   regime_features
   RegimeDetector
