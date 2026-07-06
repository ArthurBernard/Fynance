--------------------------------------
 Estimator (:mod:`fynance.estimator`)
--------------------------------------

.. currentmodule:: fynance.estimator

Maximum-likelihood volatility estimation
========================================

:func:`fit_volatility` fits a GARCH-family conditional-variance model
(GARCH / GJR-GARCH / EGARCH(1, 1), with Gaussian or standardized Student-t
innovations) on a return series by maximum likelihood. It returns a
:class:`VolatilityResult` carrying the fitted parameters and their standard
errors, the information criteria, the in-sample conditional volatility and
standardized residuals, plus closed-form / Monte-Carlo variance forecasting
(:meth:`VolatilityResult.forecast`) and model simulation
(:meth:`VolatilityResult.simulate`).

.. autosummary::
   :toctree: generated/

   fit_volatility
   VolatilityResult

The single authoritative conditional-variance recursions and log-likelihood
live in :mod:`fynance.models.econometric_models`
(:func:`~fynance.models.econometric_models.loglik_garch`); this module only
wires the optimiser, the standard errors and the forecasting / simulation on
top.

Internal ARMA / GARCH path
==========================

.. note::

   The pure-Python ARMA / GARCH parameter estimator (``estimation``) is an
   experimental placeholder and is not part of the public API. The public
   entry point for ARMA / GARCH parameters is
   :func:`~fynance.models.econometric_models.get_parameters`, documented under
   :doc:`models.econometric_models`.
