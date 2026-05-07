=======================================================
 Fynance - Machine learning tools designed for finance 
=======================================================

.. image:: https://img.shields.io/pypi/pyversions/fynance
    :alt: PyPI - Python Version
.. image:: https://img.shields.io/pypi/v/fynance.svg
    :target: https://pypi.org/project/fynance/
.. image:: https://img.shields.io/pypi/status/fynance.svg?colorB=blue
    :target: https://pypi.org/project/fynance/
.. image:: https://img.shields.io/github/license/ArthurBernard/fynance.svg
    :target: https://github.com/ArthurBernard/Fynance/blob/master/LICENSE.txt

.. image:: https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml/badge.svg?branch=master
    :target: https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml
    :alt: CI
.. image:: https://readthedocs.org/projects/fynance/badge/?version=latest
    :target: https://fynance.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/gh/ArthurBernard/Fynance/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/ArthurBernard/Fynance
    :alt: Test Coverage
.. image:: https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/badges/interrogate.svg
    :target: https://github.com/ArthurBernard/Fynance
    :alt: Docstring Coverage
.. image:: https://pepy.tech/badge/fynance
    :target: https://pepy.tech/project/fynance

- **Documentation**: http://fynance.readthedocs.io/en/latest/index.html
- **Source code**: http://github.com/ArthurBernard/Fynance

**Fynance** is a Python (and Cython) package providing **machine learning**, **econometric** and **statistical** tools designed for **financial analysis** and **backtesting of trading strategies**. The `documentation`_ is available with descriptions and examples for all public APIs.

.. _documentation: https://fynance.readthedocs.io/en/latest/index.html

The ``fynance.features`` and ``fynance.algorithms.allocation`` subpackages are stable. Other subpackages (``fynance.models``, ``fynance.backtest``) are actively developed and may evolve.

--------------
 Presentation
--------------

The ``fynance`` package contains five subpackages:

- **Algorithms** (``fynance.algorithms``) contains:
    - **Portfolio allocations** (e.g. ERC, HRP, IVP, MDP, MVP, etc.).
    - **Rolling objects** for algorithms (e.g. rolling_allocation, etc.).

- **Backtesting** objects (``fynance.backtest``) contains:
    - Module to plot profit and loss, and measure of performance.

- **Feature** tools (``fynance.features``) contains:
    - **Financial indicators** (e.g. bollinger_band, cci, hma, macd_hist, macd_line, rsi, etc.).
    - **Statistical momentums** (e.g. sma, ema, wma, smstd, emstd, wmstd, etc.).
    - **Metrics** (e.g. annual_return, annual_volatility, calmar, diversified_ratio, mdd, sharpe, z_score, etc.).
    - **Scale** (e.g. Scale object, normalize, standardize, roll_normalize, roll_standardize, etc.).
    - **Rolling functions** (e.g. roll_min, roll_max).
    - **Filters** (e.g. Kalman filter with RTS smoother and MLE parameter estimation).

- **Time-series models** (``fynance.models``) contains:
    - **Econometric models** (e.g. MA, ARMA, ARMA_GARCH, ARMAX_GARCH, etc.).
    - **Neural network models** with **PyTorch** (e.g. MultiLayerPerceptron, LSTM, MultiHeadAttention, etc.).
    - **Rolling walk-forward evaluation** for models (e.g. RollMultiLayerPerceptron, etc.).

Please refer to the `documentation`_ for more details on the tools available in the ``fynance`` package.

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

------
 Demo 
------

- **Backtest** (performance, drawdown and rolling sharpe ratio) of a **trading strategy** did with a **rolling neural network** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

.. image:: https://github.com/ArthurBernard/Fynance/blob/master/pictures/backtest_RollNeuralNet.png

- **Loss functions** and **performances** (trading strategy) of five rolling neural networks on the **training and testing period** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

.. image:: https://github.com/ArthurBernard/Fynance/blob/master/pictures/loss_RollNeuralNet.png

