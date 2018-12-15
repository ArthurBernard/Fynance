# Fynance: Machine learning tools for financial analysis 

## Description

This project contains several python (and cython) tools for financial analysis:
- Econometric models
- Neural Network
- Feature extraction method
- Financial indicators
- Backtesting
- Notebooks with some exemples
- Etc.

## Installation

Clone the repository and at the root of the folder:

> $ python setup.py build_ext --inplace   
> $ pip install fynance   

## Demo

- Backtest (performance, drawdown and rolling sharpe ratio) of a trading strategy did with a rolling neural network (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

![backtest_RollNeuralNet](https://github.com/ArthurBernard/Fynance/blob/master/pictures/backtest_RollNeuralNet.png)

- Loss functions and performances (trading strategy) of five rolling neural networks on the training and testing period (see Notebooks/Exemple_Rolling_NeuralNetwork.ipynb for more details):

![loss_RollNeuralNet](https://github.com/ArthurBernard/Fynance/blob/master/pictures/loss_RollNeuralNet.png)