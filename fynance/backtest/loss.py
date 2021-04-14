#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2021-03-27 11:24:24
# @Last modified by: ArthurBernard
# @Last modified time: 2021-04-14 11:21:35

""" Loss object. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages
from fynance.backtest.plot import PlotSeries


# TODO : Inherits of methods and properties of numpy ndarray ?


__all__ = ['LossSeries']


class LossSeries:
    """ Series of loss function values.

    Attributes
    ----------
    values : np.ndarray[np.float64, ndim=1]
        Values of the loss series.

    Methods
    -------
    set_plot
    plot

    """

    def __init__(self):
        """ Initialize loss series object. """
        self.values = np.array([])

    def append(self, other):
        """ Append value to loss series.

        Parameters
        ----------
        other : float or array_like
            These values are append to loss series object.

        """
        if isinstance(other, (float, int)):
            other = [other]

        self.values = np.append(self.values, other)

    def reset(self):
        """ Drop the stored values in the loss series. """
        self.values = np.array([])

    # def __add__(self, other):
    #    return self.append(other)

    # def __iadd__(self, other):
    #    return self.append(other)

    def __repr__(self):
        """ Represent the loss series. """
        return f"{self.__class__.__name__}({str(self.values)})"

    def __str__(self):
        """ Represent the loss series as string. """
        return str(self.values)

    def set_plot(self, ax=None, **kwargs):
        """ Instanciate plot object for the loss series.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to display backtest.
        ax : matplotlib.axes
            Axe(s) to display a part of backtest.
        **kwargs : `matplotlib.lines.Line2D` properties, optional
            *kwargs* are used to specify properties like a line label (for
            auto legends), linewidth, antialiasing, marker face color.
            Example::

            >> plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)
            >> plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')

            If you specify multiple lines with one plot call, the kwargs apply
            to all those lines. In case the label object is iterable, each
            element is used as labels for each set of data.

            For more details cf matplotlib documentation [3]_.

        References
        ----------
        .. [3] https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

        """
        # Setup plot object
        self.plot = PlotSeries(ax=ax)

        # Set ax properties
        ax.set_title('Loss function')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.grid()
        ax.set_autoscaley_on(True)
        ax.set_autoscalex_on(True)

        # Set line properties
        self.plot.plot(self.values, **kwargs)

    def update_plot(self):
        """ Update plot object with loss series data. """
        self.plot.update(self.values)


if __name__ == "__main__":
    pass
