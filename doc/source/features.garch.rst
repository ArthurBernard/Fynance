*****************
GARCH volatility
*****************

Conditional volatility as a **causal** feature, from a GARCH-family fit. The
parameters are estimated by maximum likelihood on a training prefix (optionally
refit on the expanding window every ``refit`` steps), then the conditional
volatility :math:`\sigma_t` is forward-filtered over the whole series — which is
causal because :math:`\sigma_t` is :math:`\mathcal F_{t-1}`-measurable. The first
``min_train`` values (the in-sample warmup) are returned as ``NaN``.

The ``model`` (``'garch'`` / ``'gjr'`` / ``'egarch'``) and ``dist``
(``'normal'`` / ``'t'``) arguments select the specification. The default
(``model='garch'``, ``dist='normal'``) keeps the historical ARMA-GARCH
estimation path unchanged; any non-default choice routes each block through
:func:`fynance.estimator.fit_volatility`.

The single authoritative ARMA/GARCH implementation lives in
:mod:`fynance.models.econometric_models` (the Numba recursion) and
:mod:`fynance.estimator` (the likelihood and the MLE driver); this feature only
wires the thin fit + forward-filter on top.

.. currentmodule:: fynance.features.garch

.. autosummary::
   :toctree: generated/

   garch_volatility
