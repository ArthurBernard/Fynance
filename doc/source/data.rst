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
   session_mask
   session_id
   session_bounds
   split_sessions
   train_test_split
   walk_forward
   combinatorial_purged_cv

Custom sources
==============

The registry extension API: subclass :class:`BaseDataSource`, `register` it under
a file extension, and :func:`load` (or :func:`get_source`) will dispatch to it.

.. autosummary::
   :toctree: generated/

   BaseDataSource
   register
   get_source
