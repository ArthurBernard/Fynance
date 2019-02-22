=====================================================================
Fynance - Machine learning tools for financial analysis [In Progress]
=====================================================================

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

`fynance` is a beta version of **python/cython project** [*]_ that includes several **machine learning**, **econometric** and **statistical** tools specialy adapted for **trading strategy** and **financial analysis**.

Description
===========

This project contains several **python** (and **cython**) tools for **trading strategy** and **financial analysis**:

- Neural Networks 

- Feature extraction methods    

- Financial indicators    

- Backtesting    

- Econometric models   

- Notebooks with some exemples    

- Etc.    

Please refer you to the `documentation`_ to see more details on different tools available in `fynance` package. Documentation contains some descriptions and examples for functions, classes and methods.    

.. _documentation: https://fynance.readthedocs.io/en/latest/index.html

Installation
============

From pip
--------

    $ pip install fynance

From GitHub
-----------

Use the command:

    $ git clone https://github.com/ArthurBernard/Fynance.git

    $ cd Fynance

    $ python setup.py build_ext --inplace
    
    $ python setup.py install --user

Compile from cython files   
-------------------------

If you want to compile fynance package from cython files you must edit `setup.py` file and set :: USE_CYTHON = True 

And use the commad:

    $ python setup.py build_ext --inplace    
    
    $ python setup.py install --user   


Demo
====

- **Backtest** (performance, drawdown and rolling sharpe ratio) of a **trading strategy** did with a **rolling neural network** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

.. image:: https://github.com/ArthurBernard/Fynance/blob/master/pictures/backtest_RollNeuralNet.png

- **Loss functions** and **performances** (trading strategy) of five rolling neural networks on the **training and testing period** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

.. image:: https://github.com/ArthurBernard/Fynance/blob/master/pictures/loss_RollNeuralNet.png

.. [*] Package not achieved, always in progress. All advice is welcome.