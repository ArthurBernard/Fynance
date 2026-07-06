******************************
Purged walk-forward tuning
******************************

Grid/random hyperparameter search scored on purged walk-forward folds (see
:mod:`fynance.data.split`), so configurations are compared out-of-sample only.
The trial count feeds :func:`fynance.research.guards.deflated_sharpe_ratio` to
deflate the winning configuration's Sharpe by the number of trials.

.. currentmodule:: fynance.models.tuning

.. autosummary::
   :toctree: generated/

   SearchResult
   walk_forward_search
