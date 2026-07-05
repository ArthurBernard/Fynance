**************
Cross-section
**************

Per-bar cross-sectional transforms on ``(T, N)`` panels, all NaN-aware (a
missing asset at bar ``t`` is excluded from that bar's statistic and stays
NaN in the output): rank (:func:`~fynance.features.cross_section.cs_rank`),
z-score (:func:`~fynance.features.cross_section.cs_zscore`), demeaning
(:func:`~fynance.features.cross_section.cs_demean`), winsorization
(:func:`~fynance.features.cross_section.cs_winsorize`) and OLS neutralization
against one or more exposures
(:func:`~fynance.features.cross_section.cs_neutralize`).

.. currentmodule:: fynance.features.cross_section

.. autosummary::
   :toctree: generated/

   cs_rank
   cs_zscore
   cs_demean
   cs_winsorize
   cs_neutralize
