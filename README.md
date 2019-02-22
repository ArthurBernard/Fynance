# Fynance - Machine learning tools for financial analysis [In Progress]

[![PyPI](https://img.shields.io/pypi/v/fynance.svg)](https://pypi.org/project/fynance/)
[![Status](https://img.shields.io/pypi/status/fynance.svg?colorB=blue)](https://pypi.org/project/fynance/)
[![Build Status](https://travis-ci.org/ArthurBernard/Fynance.svg?branch=master)](https://travis-ci.org/ArthurBernard/Fynance)
[![license](https://img.shields.io/github/license/ArthurBernard/fynance.svg)](https://github.com/ArthurBernard/Fynance/blob/master/LICENSE.txt)
[![Downloads](https://pepy.tech/badge/fynance)](https://pepy.tech/project/fynance)
[![Documentation Status](https://readthedocs.org/projects/fynance/badge/?version=latest)](https://fynance.readthedocs.io/en/latest/?badge=latest)

`fynance` is a beta version of **python/cython project** that includes several **machine learning**, **econometric** and **statistical** tools specialy adapted for **trading strategy** and **financial analysis**.

## Description

This project contains several **python** (and **cython**) tools for **trading strategy** and **financial analysis**:
- Econometric models
- Neural Networks
- Feature extraction methods
- Financial indicators
- Backtesting
- Notebooks with some exemples
- Etc.

Please refer you to the [documentation](https://fynance.readthedocs.io/en/latest/index.html) to see more details on different tools available in `fynance` package. Documentation contains some descriptions and examples for functions, classes and methods.    

## Installation

#### From pip

> $ pip install fynance

#### From GitHub

Use the command:

> $ git clone https://github.com/ArthurBernard/Fynance.git    
> $ cd Fynance   
> $ python setup.py build_ext --inplace    
> $ python setup.py install --user   

#### Compile from cython files   

If you want to compile fynance package from cython files you must set `USE_CYTHON = True` in `setup.py` file.    

> $ python setup.py build_ext --inplace    
> $ python setup.py install --user   

## Demo

- **Backtest** (performance, drawdown and rolling sharpe ratio) of a **trading strategy** did with a **rolling neural network** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

![backtest_RollNeuralNet](https://github.com/ArthurBernard/Fynance/blob/master/pictures/backtest_RollNeuralNet.png)

- **Loss functions** and **performances** (trading strategy) of five rolling neural networks on the **training and testing period** (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

![loss_RollNeuralNet](https://github.com/ArthurBernard/Fynance/blob/master/pictures/loss_RollNeuralNet.png)

Package not achieved, always in progress. All advice is welcome.