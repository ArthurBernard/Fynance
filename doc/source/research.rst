------------------------------------
 Research (:mod:`fynance.research`)
------------------------------------

A data-agnostic harness for running strategy experiments and emitting portable
result artifacts. fynance never stores results itself — artifacts are written to
a caller-provided ``output_dir``. Built and tested on synthetic data; real-data
adapters live downstream.

.. currentmodule:: fynance.research

.. autosummary::
   :toctree: generated/

   Experiment
   gbm
   regime_switching
