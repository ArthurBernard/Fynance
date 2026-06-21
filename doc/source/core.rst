-------------------------------
 Core (:mod:`fynance.core`)
-------------------------------

The spine of the library: the :class:`~fynance.core.PriceSeries` value object,
the :class:`~fynance.core.OHLCV` aligned multi-series container, and the
:mod:`typing.Protocol` seams the pipeline composes through.

.. currentmodule:: fynance.core

.. autosummary::
   :toctree: generated/

   PriceSeries
   OHLCV
   DataSource
   FeatureTransform
   SignalModel
   Allocator
   CostModel
   Metric
