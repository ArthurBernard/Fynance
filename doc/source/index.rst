============================
 Welcome to Fynance project 
============================

This is the documentation of ``fynance``, a **python/cython project** that includes several **machine learning**, **econometric** and **statistical** subpackages specialy adapted for **financial analysis** and **backtest trading strategies**.

--------------
 Installation 
--------------

From PyPI
=========

   $ pip install fynance

From source (GitHub)
====================

If you want to compile ``fynance`` package from cython files you must set ``USE_CYTHON = True`` in ``setup.py`` file. Otherwise set it to ``USE_CYTHON = False``. By default ``USE_CYTHON = 'auto'``.

   $ git clone https://github.com/ArthurBernard/Fynance.git

   $ cd Fynance

   $ python setup.py build_ext --inplace

   $ python setup.py install --user

--------------
 Presentation 
--------------

The ``fynance`` package contains currently five available modules:

- Algorithms :mod:`fynance.algorithms` contains:
   - Portfolio allocations :mod:`fynance.algorithms.allocation`.
   - Rolling objects :mod:`fynance.algorithms.rolling` for algorithms.

- Backtesting objects :mod:`fynance.backtest`. TODO : improve backtest module.

- Time-series models :mod:`fynance.models` contains:
   - Econometric models :mod:`fynance.models.econometric_models`.
   - Neural network models :mod:`fynance.models.neural_network` (PyTorch).
   - Rolling objects :mod:`fynance.models.rolling` for models, currently work only with neural network models.

- Neural networks :mod:`fynance.neural_networks` with Keras (backend Tensorflow or Theano).

- Tools :mod:`fynance.tools` contains:
   - Financial indicators :mod:`fynance.tools.indicators`.
   - Statistical momentums :mod:`fynance.tools.momentums`.
   - Metrics :mod:`fynance.tools.metrics`. 

----------
 Contents 
----------

.. toctree::
   :maxdepth: 2

   algorithms
   backtest
   models
   neural_networks
   tools