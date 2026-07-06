*******************
Feature engineering
*******************

Feature-engineering and selection research tools: multi-resolution feature stacking (:func:`~fynance.features.engineering.multi_resolution`), regime-adaptive windows (:func:`~fynance.features.engineering.adaptive_roll` / :func:`~fynance.features.engineering.adaptive_volatility`), fixed-width fractional differentiation (:func:`~fynance.features.engineering.fracdiff`), a Granger-causality test (:func:`~fynance.features.engineering.granger_causality`) and incremental moments (:func:`~fynance.features.engineering.IncrementalMoments`).

.. currentmodule:: fynance.features.engineering

.. autosummary::
   :toctree: generated/

   multi_resolution
   adaptive_roll
   adaptive_volatility
   fracdiff
   granger_causality
   IncrementalMoments
