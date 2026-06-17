-------------------------------------
 Strategy (:mod:`fynance.strategy`)
-------------------------------------

Optional orchestrator composing the pipeline maillons — features, model, signal
and costs — into one runnable object. Every slot is optional; the pieces stay
usable standalone.

.. currentmodule:: fynance.strategy

.. autosummary::
   :toctree: generated/

   Strategy

Multi-input (``X`` / ``y``)
===========================

By default a :class:`Strategy` builds its features from prices via the
``features`` callable. When that is not enough — exogenous, regime, or
multi-venue inputs that the price-only featurizer cannot produce — pass a
**precomputed feature matrix** ``X`` (rows aligned with the price index) to
:meth:`Strategy.run`, :meth:`Strategy.run_walk_forward`, or
:func:`fynance.research.run_experiment`. The prices then drive **P&L only**; the
model reads ``X``.

In walk-forward, the harness slices ``X[train]`` / ``X[test]`` per window — that
slice *is* the model refit (the rolling neural network). ``y`` is the supervised
target sliced the same way (for an :class:`~fynance.models.objective.ObjectiveModel`
it is the realized returns). ``X`` and ``y`` dtypes are preserved (e.g. ``float32``
stays ``float32``), and the matrix must be **causal** — every row uses only data
up to its timestamp.

.. code-block:: python

   from fynance.strategy import Strategy
   from fynance.research import run_experiment

   # X: (T, n_features) causal matrix aligned with `prices`; y: (T, 1) target.
   exp = run_experiment(strategy, prices, name="nn", X=X, y=y,
                        walk_forward={"train": 750, "test": 250})

See :doc:`research_workflow` for the end-to-end loop.
