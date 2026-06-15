.. raw:: html

   <div class="hero-header">
     <img class="only-light" src="_static/logo-light-transparent.svg" alt="Fynance logo">
     <img class="only-dark"  src="_static/logo-dark-transparent.svg"  alt="Fynance logo">
     <h1 class="hero-title">Fynance</h1>
   </div>

.. rst-class:: hidden-rst-title

============================
 Welcome to Fynance project
============================

.. raw:: html

   <div class="badge-row">
     <img src="https://img.shields.io/pypi/pyversions/fynance.svg" alt="Python versions">
     <a href="https://pypi.org/project/fynance/"><img src="https://img.shields.io/pypi/v/fynance.svg" alt="PyPI version"></a>
     <a href="https://pypi.org/project/fynance/"><img src="https://img.shields.io/pypi/status/fynance.svg?colorB=blue" alt="PyPI status"></a>
     <a href="https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml"><img src="https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
     <a href="https://github.com/ArthurBernard/Fynance/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/ArthurBernard/fynance.svg" alt="License"></a>
     <a href="https://fynance.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/fynance/badge/?version=latest" alt="Documentation"></a>
     <a href="https://codecov.io/gh/ArthurBernard/Fynance"><img src="https://codecov.io/gh/ArthurBernard/Fynance/branch/develop/graph/badge.svg" alt="Coverage"></a>
     <a href="https://github.com/ArthurBernard/Fynance"><img src="https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/badges/interrogate_badge.svg" alt="Docstring coverage"></a>
     <a href="https://pepy.tech/project/fynance"><img src="https://pepy.tech/badge/fynance" alt="Downloads"></a>
   </div>

**Fynance** is a Python and Cython package providing **machine learning**,
**econometric** and **statistical** tools for **financial analysis** and
**backtesting of trading strategies**.

.. code-block:: bash

   pip install fynance

.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 0
   :padding: 0

   .. grid-item-card:: :octicon:`goal;1.2em;sd-mr-1` Algorithms
      :link: portfolio
      :link-type: doc

      Portfolio allocation (ERC, HRP, IVP, MDP, MVP) and rolling
      walk-forward wrappers.

   .. grid-item-card:: :octicon:`rocket;1.2em;sd-mr-1` Backtest
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

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   quickstart
   changelog

.. toctree::
   :hidden:
   :caption: Reference

   core
   data
   signal
   portfolio
   backtest
   estimator
   features
   metrics
   models
   plot
   strategy
