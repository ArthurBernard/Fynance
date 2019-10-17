------------------------------------------------
 Financial tools package (:mod:`fynance.tools`) 
------------------------------------------------

.. automodule:: fynance.tools
   :no-members:
   :no-inherited-members:
   :no-special-members:

These modules contain several financial tools such as metrics and indicators, and also some statistical tools as moving averages and moving standard deviations.

Some precisions about parameters' notation in the following modules:

- ``X`` is the time-series of returns, prices or indexed values. It can be one or two-dimensional array, if ``X`` is two-dimensional then you can precise the axis along wich make the computation. By default the compuatation is done along axis 0, i.e. each row is an observation at time t and each column is a different time-series.

- ``w`` is the size of the lagged window, e.g. a simple moving average of ``X`` is noted :math:`sma^w_t(X) = \frac{1}{w} \sum^{w-i}_{i=0} X_{t-i}`.

- ``kind`` means the method to compute moving average and/or standard deviation, simple ``'s'``, weighted ``'w'`` and exponential ``'e'`` are allowed.

- ``slow_w`` and ``fast_w`` are the size of the lagged windows for respectively long and short moving averages/standard deviations.

- ``period`` is the number of period per year of data, e.g in daily data ``period=252`` trading days per year or ``period=365`` days per year, it depends of data.

- ``axis`` is the axis on which the computation is done. This parameter is relevant only for two-dimensional arrays. By default the compuatation is done along axis 0, i.e. each row is an observation at time t and each column is a different time-series.

- ``dtype`` is the type of output data in the array, only 'numerical types' are allowed (e.g. ``float``, ``double``, ``int``, ``np.float16``, etc.). By default is None, it infer the data type from ``X`` input.
