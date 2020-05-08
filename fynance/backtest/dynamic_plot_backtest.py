#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-05 19:17:04
# @Last modified by: ArthurBernard
# @Last modified time: 2020-05-08 10:08:40

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

    def __init__(self, fig=None, ax=None, size=(9, 6), **kwargs):
        """ Initialize method.

        Sets size of training and predicting period, inital value to backtest,
        a target filter and training parameters.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to display backtest.
        ax : matplotlib.axes, optional
            Axe(s) to display a part of backtest.
        size : tuple, optional
            Size of figure, default is (9, 6)
        kwargs : dict, optional
            Axes configuration, cf matplotlib documentation [1]_. Default is
            {'yscale': 'linear', 'xscale': 'linear', 'ylabel': '',
            'xlabel': '', 'title': '', 'tick_params': {}}

        References
        ----------
        .. [1] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

        """
        super(DynaPlotBackTest, self).__init__(fig, ax, size, True, **kwargs)
        self.ax_params = kwargs

    # def _set_figure(self, fig, ax, size):
    #    """ Set figure, axes and parameters for dynamic plot. """
    #    PlotBackTest._set_figure(self, fig, ax, size)
    #    plt.ion()
    #    return self

    def set_axes(self):
        """ Set axes with initial parameters. """
        self._set_axes(**self.ax_params)

    def _set_axes(self, yscale='linear', xscale='linear', ylabel='',
                  xlabel='', title='', tick_params={}):
        """ Set axes parameters. """
        self.ax.set_yscale(yscale)
        self.ax.set_xscale(xscale)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel, x=0.9)
        self.ax.set_title(title)
        self.ax.tick_params(**tick_params)

        return self
