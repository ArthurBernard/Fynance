****************
OHLCV indicators
****************

Multi-series technical indicators that need more than the close — High/Low
(:func:`~fynance.features.ohlcv.atr`, :func:`~fynance.features.ohlcv.adx`,
:func:`~fynance.features.ohlcv.williams_r`) or Volume
(:func:`~fynance.features.ohlcv.obv`, :func:`~fynance.features.ohlcv.vwap`).

Each takes the raw aligned arrays (the primary API) **or** a single
:class:`~fynance.core.OHLCV` container as the first argument. All are **causal**
(the value at ``t`` uses only ``data[..t]``); the rolling loops are Numba
``@njit`` kernels.

.. currentmodule:: fynance.features.ohlcv

.. autosummary::
   :toctree: generated/

   atr
   adx
   williams_r
   obv
   vwap
