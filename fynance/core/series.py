#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2021-04-05 18:43:52
# @Last modified by: ArthurBernard
# @Last modified time: 2021-05-01 11:44:33

""" Basis class of series objects. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages
from fynance.backtest.plot import PlotSeries

__all__ = ['Series']


class Series(np.ndarray):
    """ Subclass of numpy ndarray. """

    def __new__(cls, input_array):
        """ Create and return a new object. """
        obj = np.asarray(input_array).view(cls)
        obj.plot_series = None

        return obj

    def __array_finalize__(self, obj):
        """ None. """
        if obj is None:

            return

        self.plot_series = getattr(obj, 'plot_series', None)

    def append(self, values, axis=0):
        """ Append values to Series.

        Parameters
        ----------
        values : array_like
            These values are appended to a copy of `arr`.  It must be of the
            correct shape (the same shape as `arr`, excluding `axis`).  If
            `axis` is not specified, `values` can be any shape and will be
            flattened before use.
        axis : int, optional
            The axis along which `values` are appended.  If `axis` is None,
            both `arr` and `values` are flattened before use.

        Returns
        -------
        Series
            A copy of `arr` with `values` appended to `axis`.  Note that
            `append` does not occur in-place: a new array is allocated and
            filled.  If `axis` is None, `out` is a flattened array.

        """
        return np.append(self, values, axis=axis)

    def plot(self, ax=None, **kwargs):
        """ Instanciate a new plot object for Series and display self data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axe to display series data. If None, then a new axe is created.
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
        self.plot_series = PlotSeries(ax=ax)

        # Set line properties with kwargs parameters and plot data series
        self.plot_series(y=self, **kwargs)

    def update_plot(self):
        """ Update the plot object for Series with self data. """
        if self.plot_series is None:
            # Instanciate PlotSeries object
            self.plot()

        else:
            # Call PlotSeries object with updated data series
            self.plot_series(y=self)


if __name__ == "__main__":
    pass
