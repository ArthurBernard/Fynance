#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-05 19:17:04
# @Last modified by: ArthurBernard
# @Last modified time: 2019-05-24 12:47:17

""" Module with some function plot backtest. """

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Local packages
from fynance.backtest.plot_backtest import PlotBackTest

# Set plot style
plt.style.use('seaborn')

__all__ = ['DynaPlotBackTest']


# TODO : FINISH DOCSTRING
class DynaPlotBackTest(PlotBackTest):
    """ Dynamic plot backtest object.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure to display backtest.
    ax : matplotlib.axes
        Axe(s) to display a part of backtest.

    Methods
    -------
    plot(y, x=None, names=None, col='Blues', lw=1., **kwargs)
        Plot performances.

    See Also
    --------
    PlotBackTest, display_perf, set_text_stats

    """

    plt.ion()
    # def _set_figure(self, fig, ax, size):
    #    """ Set figure, axes and parameters for dynamic plot. """
    #    PlotBackTest._set_figure(self, fig, ax, size)
    #    plt.ion()
    #    return self

    def plot(self, y, x=None, names=None, col='Blues', lw=1., unit='raw',
             **kwargs):
        """ Dynamic plot performances.

        Parameters
        ----------
        y : np.ndarray[np.float64, ndim=2], with shape (`T`, `N`)
            Returns or indexes.
        x : np.ndarray[ndim=2], with shape (`T`, 1), optional
            x-axis, can be series of int or dates or string.
        names : str, optional
            Names y lines for legend.
        col : str, optional
            Color of palette, cf seaborn documentation [2]_.
            Default is 'Blues'.
        lw : float, optional
            Line width of lines.
        kwargs : dict, optional
            Parameters for `ax.legend` method, cf matplotlib
            documentation [3]_.

        Returns
        -------
        pbt : PlotBackTest
            Self object.

        References
        ----------
        .. [2] https://seaborn.pydata.org/api.html
        .. [3] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

        """
        # Set data
        T, N = y.shape
        if x is None:
            x = np.arange(T)

        if names is None:
            names = [r'$Model_{}$'.format(i) for i in range(N)]

        elif isinstance(names, str):
            names = [r'${}_{}$'.format(names, i) for i in range(N)]

        col = sns.color_palette(col, N)

        # Set graphs
        h = self.ax.plot(x, y, LineWidth=lw)

        # Set lines
        for i in range(N):
            h[i].set_color(col[i])
            h[i].set_label(self._set_name(names[i], y[:, i], unit=unit))

        # display
        self.ax.legend(**kwargs)
        # self.f.canvas.draw()

        return self

    def _set_name(self, name, y, unit='raw', **kwargs):
        if unit.lower() == 'raw':

            return '{}: {:.2f}'.format(name, y[-1])

        elif unit.lower() == 'perf':

            return '{}: {:.0%}'.format(name, y[-1] / y[0] - 1)

        else:

            raise ValueError
