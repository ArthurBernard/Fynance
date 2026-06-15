********************
Portfolio allocation
********************

Currently this module contains only five algorithms: Equal Risk Contribution (:func:`~fynance.portfolio.allocation.ERC`), Hierarchical Risk Parity (:func:`~fynance.portfolio.allocation.HRP`), Inverse Variance Portfolio (:func:`~fynance.portfolio.allocation.IVP`), Maximum Diversified Portfolio (:func:`~fynance.portfolio.allocation.MDP`), Minimum Variance Portfolio constrained (:func:`~fynance.portfolio.allocation.MVP`) and unconstrained (:func:`~fynance.portfolio.allocation.MVP_uc`).

The module contains also an object to roll these allocations algorithms (:func:`~fynance.portfolio.allocation.rolling_allocation`).

.. currentmodule:: fynance.portfolio.allocation

Allocation algorithms
=====================

.. autosummary::
   :toctree: generated/

   ERC
   HRP
   IVP
   MDP
   MVP
   MVP_uc

Rolling object
==============

.. autosummary::
   :toctree: generated/

   rolling_allocation
