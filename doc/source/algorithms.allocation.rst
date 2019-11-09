********************
Portfolio allocation
********************

Currently this module contains only five algorithms: Equal Risk Contribution (:func:`~fynance.algorithms.allocation.ERC`), Hierarchical Risk Parity (:func:`~fynance.algorithms.allocation.HRP`), Inverse Variance Portfolio (:func:`~fynance.algorithms.allocation.IVP`), Maximum Diversified Portfolio (:func:`~fynance.algorithms.allocation.MDP`), Minimum Variance Portfolio constrained (:func:`~fynance.algorithms.allocation.MVP`) and unconstrained (:func:`~fynance.algorithms.allocation.MVP_uc`).

The module contains also an object to roll these allocations algorithms (:func:`~fynance.algorithms.allocation.rolling_allocation`).

.. currentmodule:: fynance.algorithms.allocation

Allocation algorithms
=====================

.. autosummary::
   :toctree: generated/

   fynance.algorithms.allocation.ERC
   fynance.algorithms.allocation.HRP
   fynance.algorithms.allocation.IVP
   fynance.algorithms.allocation.MDP
   fynance.algorithms.allocation.MVP
   fynance.algorithms.allocation.MVP_uc

Rolling object
==============

.. autosummary::
   :toctree: generated/

   fynance.algorithms.allocation.rolling_allocation
