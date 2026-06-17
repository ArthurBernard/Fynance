=================
Research workflow
=================

The canonical loop for turning a strategy idea into a **reproducible experiment**
with portable, remotely-viewable results — the documented form of what the
``/run-strategy`` agent skill automates. Every step below is data-agnostic and
runs on synthetic data; the full script lives in
``examples/research_workflow.py``.

.. note::

   **fynance is the tool, not the results.** It ships only *synthetic* data
   generators (:func:`~fynance.research.gbm`,
   :func:`~fynance.research.regime_switching`) and **never stores results
   itself** — every artifact is written under an ``output_dir`` *you* provide.
   Real data and stored results belong in your own (downstream) repo.

.. warning::

   A result on synthetic data is a **plumbing check, not evidence of edge**. A
   real claim needs real data **and** a small permutation p-value **and** a
   healthy deflated Sharpe (below).

The loop
========

.. code-block:: text

   data → features (X, y) → strategy → run_experiment (walk-forward) → guardrails → report

1. Data
-------

Start from a :class:`~fynance.core.PriceSeries`. Here it is synthetic; on real
data you would build the series from your own source.

.. code-block:: python

   from fynance.research import gbm

   prices = gbm(1500, seed=7)        # a PriceSeries named "synthetic-gbm"
   train, test = 750, 250

2. Features (``X`` / ``y``)
---------------------------

Build a **causal** feature matrix ``X`` (trailing/rolling only, standardized with
**train-only** statistics) and a target ``y`` (the next-bar return). ``X`` is
aligned row-for-row with the price index; in walk-forward the harness slices
``X[train]`` / ``X[test]`` per window — that slice *is* the rolling model refit.
See :doc:`strategy` for the full ``X`` / ``y`` contract.

.. code-block:: python

   import numpy as np
   import fynance as fy

   p = prices.to_numpy()
   X = np.column_stack([fy.roc(p, 1), fy.roc(p, 5), fy.realized_volatility(p, w=20)])
   mu, sd = X[:train].mean(0), X[:train].std(0)
   sd[sd == 0] = 1.0
   X = np.nan_to_num((X - mu) / sd).astype(np.float32)

   y = np.zeros((p.shape[0], 1), dtype=np.float32)
   y[:-1, 0] = (p[1:] / p[:-1] - 1.0)      # next-bar return

3. Strategy
-----------

A :class:`~fynance.strategy.Strategy` composes the maillons. Two styles:

A **rule-based** strategy (a causal feature → a signal):

.. code-block:: python

   from fynance.strategy import Strategy
   from fynance.backtest import ProportionalCost

   baseline = Strategy(features=lambda p: fy.roc(p, 20), signal=np.sign,
                       cost=ProportionalCost(0.001))

An **objective-aligned** neural strategy — the net outputs positions and is
trained directly to maximize the Sharpe of ``positions * returns`` (not MSE on a
target), via :class:`~fynance.models.ObjectiveModel` and
:class:`~fynance.models.loss.SharpeLoss`. The signal is the identity (the model
already emits positions):

.. code-block:: python

   from fynance.models import ObjectiveModel, SharpeLoss

   model = ObjectiveModel(layers=(16, 8), loss=SharpeLoss(), epochs=60, seed=0)
   objective = Strategy(model=model, signal=lambda pos: pos,
                        cost=ProportionalCost(0.001))

4. Run the experiment
---------------------

:func:`~fynance.research.run_experiment` runs a seeded, cost-aware, walk-forward
backtest and returns a populated :class:`~fynance.research.Experiment`. Pass the
``feature_names`` / ``feature_desc`` / ``data_desc`` so the result records **what
produced it** (the provenance, surfaced in the report).

.. code-block:: python

   from fynance.research import run_experiment

   exp = run_experiment(
       objective, prices, name="sharpe_nn", X=X, y=y,
       walk_forward={"train": train, "test": test}, seed=0,
       feature_names=["roc_1", "roc_5", "realized_vol_20"],
       feature_desc="3 causal cols, train-standardized",
       data_desc="synthetic GBM (plumbing check, not real data)",
       output_dir="research_out",
   )

5. Guardrails
-------------

Never read the raw Sharpe alone.

- :func:`~fynance.research.permutation_test` — is the edge distinguishable from
  shuffled-returns noise? A weak ``p_value`` means it is not.
- :class:`~fynance.research.Ledger` + ``deflated_sharpe`` — does the Sharpe
  survive the number of strategies you have tried? A high raw Sharpe with a low
  deflated Sharpe is **overfitting**, not edge.

.. code-block:: python

   from fynance.research import Ledger, permutation_test

   perm = permutation_test(baseline, prices, n_permutations=100, seed=0)
   ledger = Ledger("research_out/ledger")
   ledger.append(exp)
   print(perm["p_value"], ledger.deflated_sharpe(exp))

6. Report
---------

:func:`~fynance.research.write_report` writes portable artifacts under
``output_dir``: a ``report.md`` (with the Provenance + Metrics tables and an
embedded tearsheet PNG, viewable on GitHub from a phone) and a re-runnable
notebook. Use :func:`~fynance.research.compare_report` to rank several candidates.

.. code-block:: python

   from fynance.research import write_report

   write_report(exp, "research_out")

Running it on real data
=======================

The example is deliberately data-agnostic. To run it for real, replace the
synthetic generator with your own :class:`~fynance.core.PriceSeries` (and build
``X`` from your data), then point ``output_dir`` at *your* results repo — fynance
writes nothing anywhere else. See :doc:`research` for the full harness reference
and :doc:`strategy` for the ``X`` / ``y`` contract.
