*******************
 Rolling Functions 
*******************

This module contains some tools to compute rolling functions, such as rolling minimum (:func:`~fynance.features.roll_functions.roll_min`) and rolling maximum (:func:`~fynance.features.roll_functions.roll_max`), and pairwise rolling statistics between two series (:func:`~fynance.features.roll_functions.roll_corr`, :func:`~fynance.features.roll_functions.roll_beta`).

.. currentmodule:: fynance.features.roll_functions

Minimum and maximum
===================

.. autosummary::
   :toctree: generated/

   roll_min
   roll_max

Pairwise statistics
===================

.. autosummary::
   :toctree: generated/

   roll_cov
   roll_corr
   roll_beta
   cross_corr
