-------------------------------
 Data (:mod:`fynance.data`)
-------------------------------

Ingestion layer: file adapters into :class:`~fynance.core.PriceSeries`,
alignment/resampling, and no-lookahead temporal splits.

.. currentmodule:: fynance.data

.. autosummary::
   :toctree: generated/

   load
   CSVSource
   ParquetSource
   align
   resample
   train_test_split
   walk_forward
