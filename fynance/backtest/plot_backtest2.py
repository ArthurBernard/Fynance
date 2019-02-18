#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Internal packages

# Set plot style
plt.style.use('seaborn')

__all__ = ['PlotPerf', 'PlotLoss']


class _PlotBase:
    """ Base class for plot backtest.
    
    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure to display backtest.
    ax : matplotlib.axes
        Axe(s) to display a part of backtest.
    
    Methods
    -------
    plot : Plot performance.
    
    """
    def __init__(self, fig=None, ax=None, size=(9, 6), dynamic=False, **kwargs):
        """ Init method sets size of training and predicting period, inital 
        value to backtest, a target filter and training parameters.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to display backtest.
        ax : matplotlib.axes, optional
            Axe(s) to display a part of backtest.
        size : tuple, optional
            Size of figure, default is (9, 6)
        dynamic : bool, optional
            If True set on interactive plot.
        kwargs : dict, optional
            Axes configuration, cf matplotlib documentation [1]_. Default is
            {'yscale': 'linear', 'xscale': 'linear', 'ylabel': '', 
            'xlabel': '', 'title': '', 'tick_params': {}}

        References
        ----------
        .. [1] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

        """
        
        # Set Figure
        self._set_figure(fig, ax, size, dynamic=dynamic)
        
        # Set axes
        self._set_axes(**kwargs)
        

    def _set_figure(self, fig, ax, size, dynamic=False):
        """ Set figure, axes and parameters for dynamic plot. """
        # Set figure and axes
        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots(1, 1, size)
        else:
            self.fig, self.ax = fig, ax
        if dynamic:
            plt.ion()
        return self


    def _plot(self, y, x=None, names=None, col='Blues', lw=1., **kwargs):
        """ Plot performances

        Parameters
        ----------
        y : np.ndarray[np.float64, ndim=2], with shape (`T`, `N`)
            Returns or indexes.
        x : np.ndarray[ndim=2], with shape (`T`, 1), optional
            x-axis, can be series of int or dates or string.
        names : str, optional
            Names y lines for legend.
        col : str, optional
            Color of palette, cf seaborn documentation [1]_. 
            Default is 'Blues'.
        lw : float, optional
            Line width of lines.
        kwargs : dict, optional
            Parameters for `ax.legend` method, cf matplotlib documentation [2]_.
        
        Returns
        -------
        pbt : PlotBackTest
            Self object.

        References
        ----------
        .. [1] https://seaborn.pydata.org/api.html
        .. [2] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

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
        l = self.ax.plot(x, y, LineWidth=lw)

        # Set lines
        for i in range(N):
            l[i].set_color(col[i])
            l[i].set_label(self._set_name(names[i], y[:, i], unit=unit))
        
        # display
        self.ax.legend(**kwargs)
        #self.f.canvas.draw()

        return self


    def _set_axes(self, yscale='linear', xscale='linear', ylabel='', xlabel='', title='', tick_params={}):
        """ Set axes parameters """
        self.ax.clear()
        self.ax.set_yscale(yscale)
        self.ax.set_xscale(xscale)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel, x=0.9)
        self.ax.set_title(title)
        self.ax.tick_params(**tick_params)
        return self


class PlotPerf(_PlotBase):
    """ """
    def _set_name(self, name, y):
        """ Set name of perf plot """
        return 'Perf {}: {:.0%}'.format(name, y[-1] / y[0] - 1)

    def plot(self,):
        """ """
        y_label = 'Perf'
        title = 'Model performance'


class PlotLoss(_PlotBase):
    """ """
    def _set_name(self, name, y):
        """ Set name of loss plot """
        return 'Loss {}: {:.2f}'.format(name, y[-1])

    def plot(self, y_loss):
        """ """
        self._plot(
            y_loss, names='Estim NN', col='BuGn', lw=2.
        )