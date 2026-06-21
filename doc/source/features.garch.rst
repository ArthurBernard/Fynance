*****************
GARCH volatility
*****************

Conditional volatility as a **causal** feature, from a GARCH(1,1) fit. The
parameters are estimated by maximum likelihood on a training prefix (optionally
refit on the expanding window every ``refit`` steps), then the conditional
volatility :math:`\sigma_t` is forward-filtered over the whole series — which is
causal because :math:`\sigma_t` is :math:`\mathcal F_{t-1}`-measurable. The first
``min_train`` values (the in-sample warmup) are returned as ``NaN``.

The single authoritative ARMA/GARCH implementation lives in
:mod:`fynance.models.econometric_models` (the Numba recursion) and
:mod:`fynance.estimator` (the likelihood); this feature only wires the thin
fit + forward-filter on top.

.. currentmodule:: fynance.features.garch

.. autosummary::
   :toctree: generated/

   garch_volatility
