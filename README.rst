=======================================================
 Fynance - Machine learning tools designed for finance 
=======================================================

.. image:: https://img.shields.io/pypi/pyversions/fynance
    :alt: PyPI - Python Version
.. image:: https://img.shields.io/pypi/v/fynance.svg
    :target: https://pypi.org/project/fynance/
.. image:: https://img.shields.io/pypi/status/fynance.svg?colorB=blue
    :target: https://pypi.org/project/fynance/
.. image:: https://travis-ci.org/ArthurBernard/Fynance.svg?branch=master
    :target: https://travis-ci.org/ArthurBernard/Fynance
.. image:: https://img.shields.io/github/license/ArthurBernard/fynance.svg
    :target: https://github.com/ArthurBernard/Fynance/blob/master/LICENSE.txt
.. image:: https://pepy.tech/badge/fynance 
    :target: https://pepy.tech/project/fynance
.. image:: https://readthedocs.org/projects/fynance/badge/?version=latest
    :target: https://fynance.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/lgtm/grade/python/g/ArthurBernard/Fynance.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/ArthurBernard/Fynance/context:python)
    :alt: Language grade: Python

- **Documentation**: http://fynance.readthedocs.io/en/latest/index.html
- **Source code**: http://github.com/ArthurBernard/Fynance

**Fynance** is Python (and Cython) package, it provides **machine learning**, **econometric** and **statistical** tools designed for **financial analysis** and **backtest of trading strategy**. The `documentation`_ is available with some **examples** of the use of functions and objects.

.. _documentation: https://fynance.readthedocs.io/en/latest/index.html

Currently the project is always at a **beta level**. But some parts of the project can be considered as stable, such as ``fynance.features`` (this subpackage is already coded in **Cython** to be time-efficient), ``fynance.algorithms.allocation`` (this subpackage seems stable but have to be cleaned and write in Cython), and the other subpackages are always in progress (subject to deep modification).

--------------
 Presentation 
--------------

The ``fynance`` package contains currently five subpackages:

- **Algorithms** (``fynance.algorithms``) contains:
    - **Portfolio allocations** (e.g. ERC, HRP, IVP, MDP, MVP, etc.).
    - **Rolling objects** for algorithms (e.g. rolling_allocation, etc.).

- **Backtesting** objects (``fynance.backtest``) contains:
    - Module to plot profit and loss, and measure of performance.

- **Feature** tools (``fynance.features``) contains:
    - **Financial indicators** (e.g. bollinger_band, cci, hma, macd_hist, macd_line, rsi, etc.).
    - **Statistical momentums** (e.g. sma, ema, wma, smstd, emstd wmstd, etc.).
    - **Metrics** (e.g. annual_return, annual_volatility, calmar, diversified_ratio, mdd, sharpe, z_score, etc.).
    - **Scale** (e.g. Scale object, normalize, standardize, roll_normalize, roll_standardize, etc.).
    - **Rolling functions** (e.g. roll_min, roll_max).

- **Time-series models** (``fynance.models``) contains:
    - **Econometric models** (e.g. MA, ARMA, ARMA_GARCH and ARMAX_GARCH, etc.).
    - **Neural network models** with **PyTorch** (e.g. MultiLayerPerceptron, etc.).
    - **Rolling objects** for models, currently work only with neural network models (e.g. \_RollingBasis, RollMultiLayerPerceptron, etc.).

- **Neural networks** (``fynance.neural_networks``) with **Keras** (backend **Tensorflow** or **Theano**) contains:
    - Rolling neural network models.

Please refer you to the `documentation`_ to see more details on different tools available in `fynance` package. Documentation contains some descriptions and examples for functions, classes and methods.    

.. _documentation: https://fynance.readthedocs.io/en/latest/index.html

--------------
 Installation 
--------------

From PyPI
=========

.. code-block:: console

   $ pip install fynance

From source (GitHub)
====================

If you want to compile ``fynance`` package from cython files you must set ``USE_CYTHON=True`` in ``setup.py`` file. Otherwise set it to ``USE_CYTHON=False``. By default ``USE_CYTHON='auto'``.

.. code-block:: console

   $ git clone https://github.com/ArthurBernard/Fynance.git
   $ cd Fynance
   $ python setup.py build_ext --inplace
   $ python setup.py install --user

------
 Demo 
------

- **Backtest** (performance, drawdown and rolling sharpe ratio) of a **trading strategy** did with a **rolling neural network** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

.. image:: https://github.com/ArthurBernard/Fynance/blob/master/pictures/backtest_RollNeuralNet.png

- **Loss functions** and **performances** (trading strategy) of five rolling neural networks on the **training and testing period** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

.. image:: https://github.com/ArthurBernard/Fynance/blob/master/pictures/loss_RollNeuralNet.png

.. [*] Package not achieved, always in progress. All advice is welcome.