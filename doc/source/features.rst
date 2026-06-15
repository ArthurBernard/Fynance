------------------------------------
 Features (:mod:`fynance.features`)
------------------------------------

Financial features: metrics, indicators, scaling, moving averages,
moving standard deviations and rolling functions.

.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 0
   :padding: 0

   .. grid-item-card:: :octicon:`filter;1.2em;sd-mr-1` Filters
      :link: features.filters
      :link-type: doc

      Kalman filter with RTS smoother and maximum-likelihood
      parameter estimation.

   .. grid-item-card:: :octicon:`pulse;1.2em;sd-mr-1` Indicators
      :link: features.indicators
      :link-type: doc

      Bollinger Band, CCI, Hull Moving Average, MACD, RSI.

   .. grid-item-card:: :octicon:`meter;1.2em;sd-mr-1` Statistics
      :link: features.stats
      :link-type: doc

      Accuracy and directional accuracy, positive-return share, tail
      ratio, z-score and mean absolute deviation.

   .. grid-item-card:: :octicon:`arrow-up-right;1.2em;sd-mr-1` Momentums
      :link: features.momentums
      :link-type: doc

      Simple, exponential and weighted moving averages and standard
      deviations.

   .. grid-item-card:: :octicon:`sync;1.2em;sd-mr-1` Rolling functions
      :link: features.roll_functions
      :link-type: doc

      Rolling minimum and rolling maximum.

   .. grid-item-card:: :octicon:`beaker;1.2em;sd-mr-1` Feature engineering
      :link: features.engineering
      :link-type: doc

      Multi-resolution stacking, Granger-causality filter, incremental moments.

   .. grid-item-card:: :octicon:`broadcast;1.2em;sd-mr-1` Market regime
      :link: features.regime
      :link-type: doc

      Unsupervised regime labelling by clustering volatility/return features.

   .. grid-item-card:: :octicon:`arrow-switch;1.2em;sd-mr-1` Scale
      :link: features.scale
      :link-type: doc

      Standardization, normalization and their rolling versions.

Notation
========

Common parameters across modules:

- ``X`` — time-series of returns, prices or indexed values. One- or
  two-dimensional. For 2D arrays, ``axis=0`` (default) means each row
  is an observation at time ``t`` and each column is a different
  time-series.
- ``w`` — size of the lagged window, e.g. a simple moving average of
  ``X`` is noted :math:`sma^w_t(X) = \frac{1}{w} \sum^{w-i}_{i=0} X_{t-i}`.
- ``kind`` — method for moving average and/or standard deviation:
  simple (``'s'``), weighted (``'w'``) or exponential (``'e'``).
- ``slow_w``, ``fast_w`` — size of the lagged windows for long and
  short moving averages/standard deviations.
- ``period`` — number of periods per year (e.g. ``252`` for daily
  trading days, ``365`` for daily calendar).
- ``axis`` — axis along which the computation is performed (relevant
  only for 2D arrays).
- ``dtype`` — output data type. Only numerical types are allowed
  (``float``, ``double``, ``int``, ``np.float16``, etc.). Default is
  ``None`` (inferred from ``X``).

.. toctree::
   :maxdepth: 1
   :hidden:

   features.engineering
   features.filters
   features.indicators
   features.stats
   features.momentums
   features.regime
   features.roll_functions
   features.scale
