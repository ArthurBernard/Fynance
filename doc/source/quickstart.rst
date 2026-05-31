==========
Quickstart
==========

See :doc:`installation` for prerequisites and install instructions.

----

Financial metrics
-----------------

Compute performance metrics on a return series:

.. code-block:: python

   import numpy as np
   import fynance as fy

   returns = np.random.randn(252) * 0.01   # daily returns, one year

   print(fy.sharpe(returns))               # annualised Sharpe ratio
   print(fy.calmar(returns))               # Calmar ratio
   print(fy.mdd(returns))                  # maximum drawdown
   print(fy.annual_return(returns))        # annualised return
   print(fy.annual_volatility(returns))    # annualised volatility

All metrics also have rolling variants (``roll_sharpe``, ``roll_mdd``, …)
that return a time series rather than a scalar:

.. code-block:: python

   roll_sr = fy.roll_sharpe(returns, w=63)  # 63-day rolling Sharpe
   roll_dd = fy.roll_drawdown(returns)       # rolling drawdown curve

See :doc:`features` for the full list of metrics, indicators, and momentums.

----

Technical indicators and momentums
-----------------------------------

.. code-block:: python

   prices = np.cumprod(1 + returns)

   sma   = fy.sma(prices, w=20)         # 20-day simple moving average
   ema   = fy.ema(prices, w=20)         # exponential moving average
   rsi   = fy.rsi(prices, w=14)         # RSI oscillator
   upper, mid, lower = fy.bollinger_band(prices, w=20, n=2)

----

Portfolio allocation
---------------------

Build risk-parity and minimum-variance portfolios from a covariance matrix:

.. code-block:: python

   from fynance.algorithms import ERC, HRP, MVP

   cov = np.cov(np.random.randn(5, 252))   # 5-asset covariance

   w_erc = ERC(cov)()    # Equal Risk Contribution
   w_hrp = HRP(cov)()    # Hierarchical Risk Parity
   w_mvp = MVP(cov)()    # Minimum Variance Portfolio

   print(w_erc.round(4))

See :doc:`algorithms` for all allocation methods and the rolling wrapper.

----

Walk-forward neural network
----------------------------

Train and evaluate a multi-layer perceptron over a rolling walk-forward window:

.. code-block:: python

   from fynance.models.rolling import RollMultiLayerPerceptron

   X = np.random.randn(500, 10).astype(np.float32)   # features
   y = np.random.randn(500, 1).astype(np.float32)    # targets

   model = RollMultiLayerPerceptron(
       X, y,
       layers=[64, 32],
       activation='ReLU',
   )
   # 252-day train window, 21-day test, 21-day roll step
   model(n=252, s=21, r=21)

   for result in model:
       print(result.perf_train, result.perf_test)

See :doc:`models` for LSTM, GRU, MultiHeadAttention, and custom loss functions.

----

Custom loss functions (PyTorch)
---------------------------------

Replace MSE with a financial loss to optimise Sharpe or directional accuracy
directly during training:

.. code-block:: python

   from fynance.models.loss import SharpeLoss, SortinoLoss, DirectionalAccuracyLoss

   criterion = SharpeLoss(period=252)   # maximise annualised Sharpe

   # Drop-in replacement for torch.nn.MSELoss inside any training loop:
   loss = criterion(y_pred, y_true)
   loss.backward()

See :doc:`models` for the full loss API and base class :class:`~fynance.models.loss.BaseLoss`.
