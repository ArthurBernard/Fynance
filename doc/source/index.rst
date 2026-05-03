============================
 Welcome to Fynance project 
============================

- **Source code**: http://github.com/ArthurBernard/Fynance
- **Documentation**: http://fynance.readthedocs.io/en/latest/index.html

**Fynance** is Python (and Cython) package, it provides **machine learning**, **econometric** and **statistical** tools designed for **financial analysis** and **backtest of trading strategy**.

Currently the project is always at a **beta level**. But some parts of the project can be considered as stable, such as ``fynance.features`` (this subpackage is already coded in **Cython** to be time-efficient), ``fynance.algorithms.allocation`` (this subpackage seems stable but have to be cleaned and write in Cython), and the other subpackages are always in progress (subject to deep modifications).

--------------
 Presentation 
--------------

The ``fynance`` package contains six subpackages:

- **Algorithms** (:mod:`fynance.algorithms`) contains:
   - **Portfolio allocations** (e.g. :func:`~fynance.algorithms.allocation.ERC`, :func:`~fynance.algorithms.allocation.HRP`, :func:`~fynance.algorithms.allocation.IVP`, :func:`~fynance.algorithms.allocation.MDP`, :func:`~fynance.algorithms.allocation.MVP`, etc.).
   - **Rolling objects** for algorithms (e.g. :func:`~fynance.algorithms.allocation.rolling_allocation`, etc.).

- **Backtesting** objects (:mod:`fynance.backtest`).
   - Module to plot profit and loss, and measure of performance.

- **Estimator** (:mod:`fynance.estimator`): Cython ARMA/GARCH parameter estimation, exposed via :mod:`fynance.models.econometric_models`.

- **Feature** tools (:mod:`fynance.features`) contains:
   - **Kalman filter** and RTS smoother (e.g. :func:`~fynance.features.filters.kalman_filter`, :func:`~fynance.features.filters.rts_smoother`, :func:`~fynance.features.filters.fit_kalman`).
   - **Financial indicators** (e.g. :func:`~fynance.features.indicators.bollinger_band`, :func:`~fynance.features.indicators.cci`, :func:`~fynance.features.indicators.hma`, :func:`~fynance.features.indicators.macd_hist`, :func:`~fynance.features.indicators.rsi`, etc.).
   - **Statistical momentums** (e.g. :func:`~fynance.features.momentums.sma`, :func:`~fynance.features.momentums.ema`, :func:`~fynance.features.momentums.wma`, etc.).
   - **Metrics** (e.g. :func:`~fynance.features.metrics.annual_return`, :func:`~fynance.features.metrics.sharpe`, :func:`~fynance.features.metrics.mdd`, etc.).
   - **Scale** (e.g. :func:`~fynance.features.scale.normalize`, :func:`~fynance.features.scale.standardize`, etc.).
   - **Rolling functions** (e.g. :func:`~fynance.features.roll_functions.roll_min`, :func:`~fynance.features.roll_functions.roll_max`).

- **Time-series models** (:mod:`fynance.models`) contains:
   - **Econometric models** (e.g. :func:`~fynance.models.econometric_models.MA`, :func:`~fynance.models.econometric_models.ARMA`, :func:`~fynance.models.econometric_models.ARMA_GARCH`, etc.).
   - **Neural network models** with **PyTorch**: MLP, RNN, GRU, LSTM, MultiHeadAttention.
   - **Rolling walk-forward** evaluation (e.g. :func:`~fynance.models.rolling.RollMultiLayerPerceptron`).

--------------
 Installation 
--------------

From PyPI
=========

.. code-block:: console

   $ pip install fynance

From source (GitHub)
====================

.. code-block:: console

   $ git clone https://github.com/ArthurBernard/Fynance.git
   $ cd Fynance
   $ pip install -e ".[dev]"
   $ python setup.py build_ext --inplace

----------
 Contents 
----------

.. toctree::
   :maxdepth: 2

   algorithms
   backtest
   estimator
   features
   models