************
 Statistics
************

Descriptive and normalization statistics used as features: accuracy and
directional accuracy, positive-return share, tail ratio, rolling/standard
z-score, and mean absolute deviation.

.. currentmodule:: fynance.features.stats

.. autosummary::
   :toctree: generated/

   accuracy
   directional_accuracy
   percent_positive
   tail_ratio
   z_score
   roll_z_score
   mad
   roll_mad

Money management
================

Volatility-targeted position sizing as a **causal** feature: the
:func:`~fynance.features.money_management.iso_vol` leverage scales an exposure so
its realized volatility tracks a target, using only the past.

.. currentmodule:: fynance.features.money_management

.. autosummary::
   :toctree: generated/

   iso_vol
