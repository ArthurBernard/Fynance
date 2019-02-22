Financial tools package
=======================
Several financial tools as metrics, indicators, etc.

Usage informations
------------------
Some precisions on parameters in following modules:
- `series` means the time-series of price or index values for an asset or strategy.

- `returns` means the time-series of returns for one period for an asset or strategy such that :math:`r_t = p_t - p_{t-1}`.

- `lags` means the size of the window but in the past.

- `kind_ma` means the method to compute moving average, simple, weighted and exponential are allowed.

- `slow_ma` and `fast_ma` means the size of windows for respectively long and short moving averages.

- `period` means the number of period per year of data, e.g in daily data `period = 252` trading days per year or `period = 365` days per year, it depends of data.

.. toctree::
    
    tools.indicators
    tools.metrics
    tools.momentums