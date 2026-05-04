============================
 Welcome to Fynance project
============================

**Fynance** is a Python and Cython package providing **machine learning**,
**econometric** and **statistical** tools for **financial analysis** and
**backtesting of trading strategies**.

- **Source code**: http://github.com/ArthurBernard/Fynance
- **Documentation**: http://fynance.readthedocs.io/en/latest/index.html

The project is currently at a **beta level**. Some subpackages are stable
(``fynance.features``, ``fynance.algorithms.allocation``); others are in
active development.

--------------
 Subpackages
--------------

.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 0
   :padding: 0

   .. grid-item-card:: :octicon:`graph;1.2em;sd-mr-1` Algorithms
      :link: algorithms
      :link-type: doc

      Portfolio allocation (ERC, HRP, IVP, MDP, MVP) and rolling
      walk-forward wrappers.

   .. grid-item-card:: :octicon:`gear;1.2em;sd-mr-1` Backtest
      :link: backtest
      :link-type: doc

      Profit-and-loss plotting and performance measurement.

   .. grid-item-card:: :octicon:`pulse;1.2em;sd-mr-1` Estimator
      :link: estimator
      :link-type: doc

      Cython ARMA / GARCH parameter estimation.

   .. grid-item-card:: :octicon:`stack;1.2em;sd-mr-1` Features
      :link: features
      :link-type: doc

      Kalman filter, indicators, momentums, metrics, scaling and
      rolling functions.

   .. grid-item-card:: :octicon:`workflow;1.2em;sd-mr-1` Models
      :link: models
      :link-type: doc

      Econometric models, neural networks (MLP, RNN, GRU, LSTM,
      attention) and walk-forward training.

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
