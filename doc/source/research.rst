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
   Ledger
   run_experiment
   write_report
   compare_report
   leaderboard
   permutation_test
   probabilistic_sharpe_ratio
   deflated_sharpe_ratio
   ImportanceResult
   walk_forward_mda
   gbm
   regime_switching
