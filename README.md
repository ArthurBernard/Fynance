<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/doc/source/_static/logo-dark-transparent.svg">
  <img alt="Fynance logo" src="https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/doc/source/_static/logo-light-transparent.svg" height="180px" align="left">
</picture>

# **Fynance**

[![Python versions](https://img.shields.io/pypi/pyversions/fynance)](https://pypi.org/project/fynance/)
[![PyPI](https://img.shields.io/pypi/v/fynance.svg)](https://pypi.org/project/fynance/)
[![PyPI status](https://img.shields.io/pypi/status/fynance.svg?colorB=blue)](https://pypi.org/project/fynance/)
[![CI](https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml/badge.svg)](https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ArthurBernard/fynance.svg)](https://github.com/ArthurBernard/Fynance/blob/master/LICENSE.txt)<br>
[![Documentation](https://readthedocs.org/projects/fynance/badge/?version=latest)](https://fynance.readthedocs.io/en/latest/)
[![Coverage](https://codecov.io/gh/ArthurBernard/Fynance/branch/develop/graph/badge.svg)](https://codecov.io/gh/ArthurBernard/Fynance)
[![Docstring coverage](https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/badges/interrogate_badge.svg)](https://github.com/ArthurBernard/Fynance)
[![Downloads](https://pepy.tech/badge/fynance)](https://pepy.tech/project/fynance)

___

Python and Cython package providing **machine learning**, **econometric** and **statistical** tools
for **financial analysis** and **backtesting of trading strategies**.

## Installation

```bash
pip install fynance
```

From source:

```bash
git clone https://github.com/ArthurBernard/Fynance.git
cd Fynance
pip install -e ".[dev]"
python setup.py build_ext --inplace
```

## Subpackages

**Algorithms** `fynance.algorithms`  
Portfolio allocation methods (ERC, HRP, IVP, MDP, MVP) and rolling walk-forward wrappers.

**Backtest** `fynance.backtest`  
Profit-and-loss plotting and performance measurement.

**Estimator** `fynance.estimator`  
Cython ARMA / GARCH parameter estimation.

**Features** `fynance.features`  
Kalman filter, financial indicators (Bollinger, RSI, MACD, …), statistical momentums (SMA, EMA, WMA, …),
metrics (Sharpe, Sortino, Calmar, drawdown, …), scaling, and rolling functions.

**Models** `fynance.models`  
Econometric models (MA, ARMA, ARMA-GARCH), neural networks with PyTorch (MLP, RNN, GRU, LSTM,
MultiHeadAttention), differentiable loss functions (SharpeLoss, SortinoLoss,
DirectionalAccuracyLoss), and walk-forward rolling evaluation.

## Quick start

```python
import numpy as np
import fynance as fy

# Sharpe ratio
returns = np.random.randn(252) * 0.01
print(fy.sharpe(returns))

# ERC portfolio allocation
cov = np.cov(np.random.randn(5, 252))
weights = fy.ERC(cov)
print(weights)
```

Rolling walk-forward training with a neural network:

```python
import torch
import torch.nn as nn
from fynance.models.rolling import RollMultiLayerPerceptron

model = RollMultiLayerPerceptron(X, y, layers=[64, 32])
model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
model(train_period=252, test_period=21, roll_period=21)  # walk-forward windows
for eval_set, test_set in model:   # each step trains on the past, tests the next
    model._training()
```

See [`Notebooks/pytorch_examples.ipynb`](Notebooks/pytorch_examples.ipynb) for a
runnable tour (metrics, allocation, MLP/TCN/Transformer with custom losses,
walk-forward CV).

## Links

- PyPI: https://pypi.org/project/fynance/
- Documentation: https://fynance.readthedocs.io/en/latest/
- Source: https://github.com/ArthurBernard/Fynance
- Changelog: https://github.com/ArthurBernard/Fynance/blob/master/CHANGELOG.md
