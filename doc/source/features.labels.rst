******
Labels
******

.. warning::

   These functions look at **future** prices by design -- they build
   supervised-learning *targets* (``y``), never features (``X``). Route
   their output through a purged / embargoed split (:mod:`fynance.data.split`)
   before training; never feed a label back into a model as an input.

The AFML (Lopez de Prado, *Advances in Financial Machine Learning*) labeling
stack: path-dependent triple-barrier labels (:func:`~fynance.features.labels.triple_barrier`),
meta-labels for a secondary bet-sizing model (:func:`~fynance.features.labels.meta_labels`),
and overlap-aware sample weights (:func:`~fynance.features.labels.label_concurrency`,
:func:`~fynance.features.labels.uniqueness_weights`).

.. currentmodule:: fynance.features.labels

Triple-barrier labeling
========================

.. autosummary::
   :toctree: generated/

   triple_barrier
   meta_labels

Overlap-aware sample weights
=============================

.. autosummary::
   :toctree: generated/

   label_concurrency
   uniqueness_weights
