============================
 Welcome to Fynance project 
============================

This is the documentation of ``fynance``, a **Python package** that includes several **machine learning**, **econometric** and **statistical** subpackages specialy adapted to **backtest financial analysis**, **portfolio allocation** and **backtest trading strategies**.

--------------
 Installation 
--------------

From PyPI
=========

   $ pip install fynance

From source (GitHub)
====================

If you want to compile ``fynance`` package from cython files you must set ``USE_CYTHON=True`` in ``setup.py`` file. Otherwise set it to ``USE_CYTHON=False``. By default ``USE_CYTHON='auto'``.

   $ git clone https://github.com/ArthurBernard/Fynance.git

   $ cd Fynance

   $ python setup.py build_ext --inplace

   $ python setup.py install --user

--------------
 Presentation 
--------------

The ``fynance`` package contains currently five subpackages:

- **Algorithms** (:mod:`fynance.algorithms`) contains:
   - **Portfolio allocations** (e.g. :func:`~fynance.algorithms.allocation.ERC`, :func:`~fynance.algorithms.allocation.HRP`, :func:`~fynance.algorithms.allocation.IVP`, :func:`~fynance.algorithms.allocation.MDP`, :func:`~fynance.algorithms.allocation.MVP`, etc.).
   - **Rolling objects** for algorithms (e.g. :func:`~fynance.algorithms.allocation.rolling_allocation`, etc.).

- **Backtesting** objects (:mod:`fynance.backtest`).
   - Module to plot profit and loss, and measure of performance.

- **Time-series models** (:mod:`fynance.models`) contains:
   - **Econometric models** (e.g. :func:`~fynance.models.econometric_models.MA`, :func:`~fynance.models.econometric_models.ARMA`, :func:`~fynance.models.econometric_models.ARMA_GARCH` and :func:`~fynance.models.econometric_models.ARMAX_GARCH`, etc.).
   - **Neural network models** with **PyTorch** (e.g. :func:`~fynance.models.neural_network.MultiLayerPerceptron`, etc.).
   - **Rolling objects** for models, currently work only with neural network models (e.g. :func:`~fynance.models.rolling._RollingBasis`, :func:`~fynance.models.rolling.RollMultiLayerPerceptron`, etc.).

- **Neural networks** (:mod:`fynance.neural_networks`) with **Keras** (backend **Tensorflow** or **Theano**).
   - Rolling neural network models.

- **Feature** tools (:mod:`fynance.features`) contains:
   - **Financial indicators** (e.g. :func:`~fynance.features.indicators.bollinger_band`, :func:`~fynance.features.indicators.cci`, :func:`~fynance.features.indicators.hma`, :func:`~fynance.features.indicators.macd_hist`, :func:`~fynance.features.indicators.macd_line`, :func:`~fynance.features.indicators.rsi`, etc.).
   - **Statistical momentums** (e.g. :func:`~fynance.features.momentums.sma`, :func:`~fynance.features.momentums.ema`, :func:`~fynance.features.momentums.wma`, :func:`~fynance.features.momentums.smstd`, :func:`~fynance.features.momentums.emstd` :func:`~fynance.features.momentums.wmstd`, etc.).
   - **Metrics** (e.g. :func:`~fynance.features.metrics.annual_return`, :func:`~fynance.features.metrics.annual_volatility`, :func:`~fynance.features.metrics.calmar`, :func:`~fynance.features.metrics.diversified_ratio`, :func:`~fynance.features.metrics.mdd`, :func:`~fynance.features.metrics.sharpe`, :func:`~fynance.features.metrics.z_score`, etc.).

----------
 Contents 
----------

.. toctree::
   :maxdepth: 2

   algorithms
   backtest
   models
   neural_networks
   features