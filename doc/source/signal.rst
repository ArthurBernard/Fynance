-----------------------------------
 Signal (:mod:`fynance.signal`)
-----------------------------------

The bridge from predictions to positions: mappers and a model+mapper pipeline.

The **anti-churn** mappers (:func:`ema_smooth`, :func:`deadband`,
:func:`min_hold`) are stateful but strictly causal; compose them on top of a
position to cut turnover where transaction costs would otherwise dominate (high
fees / high frequency). They pair with the train-time turnover penalty in
:class:`~fynance.models.ObjectiveModel` (its ``cost`` argument).

.. currentmodule:: fynance.signal

.. autosummary::
   :toctree: generated/

   sign
   threshold
   rank
   vol_target_position
   ema_smooth
   deadband
   min_hold
   SignalPipeline
