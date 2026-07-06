***********
Uncertainty
***********

Predictive-uncertainty wrappers around ``SignalModel``-conforming nets:
:class:`~fynance.models.uncertainty.DeepEnsemble` (member disagreement across
independently initialized nets) and :class:`~fynance.models.uncertainty.MCDropout`
(Monte Carlo Dropout kept active at inference). Both add ``predict_std`` on top
of the usual ``fit``/``predict``.

.. currentmodule:: fynance.models.uncertainty

.. autosummary::
   :toctree: generated/

   DeepEnsemble
   MCDropout
