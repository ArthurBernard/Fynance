-------------------------------
 Core (:mod:`fynance.core`)
-------------------------------

The spine of the library: the :class:`~fynance.core.PriceSeries` value object,
the :class:`~fynance.core.OHLCV` aligned multi-series container, the
:mod:`typing.Protocol` seams the pipeline composes through, and two executable
house-rule checks for them.

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

Checks
======

:func:`check_conforms` smoke-runs a protocol's methods on a candidate
instance with seeded synthetic data; :func:`assert_causal` probes an
arbitrary callable for lookahead bias by perturbing its input strictly after
a probe index and requiring the output strictly before it to be unchanged.
Both are usable straight from downstream ``pytest`` suites.

.. autosummary::
   :toctree: generated/

   check_conforms
   assert_causal
