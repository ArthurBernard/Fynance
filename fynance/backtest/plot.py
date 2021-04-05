#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2021-03-27 10:44:32
# @Last modified by: ArthurBernard
# @Last modified time: 2021-04-05 21:02:25

""" Plot objects. """

# Built-in packages

# Third party packages
import numpy as np
from matplotlib import pyplot as plt

# Local packages


__all__ = ["PlotSeries"]


# TODO : Allow Pandas and PyTorch objects


class PlotSeries:
    """ Plot object.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure to display backtest.
    ax : matplotlib.axes.Axes
        Axe(s) to display a part of backtest.
    h : matplotlib.lines.Line2D
        Line of representing the plotted data object.

    Methods
    -------
    plot
    update
    set_axes ??

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axe to display series data. If None, then a new axe is created.

    See Also
    --------
    DynaPlotBackTest, display_perf, set_text_stats

    """

    def __init__(self, ax=None):
        """ Initialize method. """
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(16, 10))

        self.ax = ax

    def __call__(self, y, x=None, append=False, **kwargs):
        """ Plot data or update the data to plot. """
        if not hasattr(self, 'h'):
            self.plot(y, x=x, **kwargs)

        else:
            self.update(y, x=x, append=append)

    def plot(self, y, x=None, **kwargs):
        """ Plot performances, loss function or any data.

        Parameters
        ----------
        y : np.ndarray[np.float64, ndim=1], with shape (`T`,)
            Returns, indexes, time-series or any data to plot.
        x : np.ndarray[ndim=1], with shape (`T`,), optional
            x-axis, can be series of integers, dates or string. If `x` is let
            to None, then range from 0 to `T` is used.
        **kwargs : `matplotlib.lines.Line2D` properties, optional
            *kwargs* are used to specify properties like a line label (for
            auto legends), linewidth, antialiasing, marker face color.
            Example::

            >>> plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)
            >>> plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')

            If you specify multiple lines with one plot call, the kwargs apply
            to all those lines. In case the label object is iterable, each
            element is used as labels for each set of data.

            For more details cf matplotlib documentation [3]_.

        Returns
        -------
        pbt : PlotBackTest
            Self object.

        References
        ----------
        .. [3] https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

        """
        # Check data
        if len(y.shape) == 1:
            T = y.size

        else:
            raise ValueError(f"y must be a 1-dimensional array")

        if x is None:
            x = np.arange(T)

        # Set graphs
        self.h = self.ax.plot(x, y, **kwargs)[0]

        return self

    def _set_axes(self, yscale='linear', xscale='linear', ylabel='',
                  xlabel='', title='', tick_params={}):
        """ Set axes parameters. """
        self.ax.clear()
        self.ax.set_yscale(yscale)
        self.ax.set_xscale(xscale)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel, x=0.9)
        self.ax.set_title(title)
        self.ax.tick_params(**tick_params)

        return self

    def update(self, y, x=None, append=False):
        """ Update plot object with data.

        Parameters
        ----------
        y : np.ndarray[np.float64, ndim=1], with shape (`T`,)
            Returns, indexes, time-series or any data.
        x : np.ndarray[ndim=1], with shape (`T`,), optional
            x-axis, can be series of integers, dates or string.
        append : bool, optional
            If True then increment the x-data and y-data with the `x` and `y`
            argument, otherwise reset the x-data and y-data.

        Returns
        -------
        PlotBackTest
            Self object.

        """
        # Set data
        if len(y.shape) == 1:
            T = y.size

        else:
            raise ValueError(f"y must be a 1-dimensional array")

        if append:
            _x = self.h.get_xdata()
            if x is None:
                x = np.arange(_x[-1] + 1, _x[-1] + T + 1)

            y = np.append(self.h.get_ydata, y)
            x = np.append(_x, x)

        elif x is None:
            x = np.arange(T)

        self.h.set_ydata(y)
        self.h.set_xdata(x)

        return self


class _MultiPlot(PlotSeries):
    def plot(self, y, x=None, **kwargs):
        if len(y.shape) != 2:
            raise ValueError("y must be a two-dimensional array.")

        T, N = y.shape



if __name__ == "__main__":
    pass
