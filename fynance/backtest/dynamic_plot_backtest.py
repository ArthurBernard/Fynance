#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Internal packages
from fynance.backtest.plot_backtest import PlotBackTest

# Set plot style
plt.style.use('seaborn')

__all__ = ['DynaPlotBackTest']

# TODO: FINISH DOCSTRING

class DynaPlotBackTest(PlotBackTest):
    """ Plot dynamically backtest.

    Attribute
    ---------
    :fig: matplotlib.figure.Figure
        Figure to display backtest.
    :ax: matplotlib.axes
        Axe(s) to display a part of backtest.

    Methods
    -------
    :plot: plot
    
    """
    def _set_figure(self, fig, ax, size):
        """ Set figure, axes and parameters for dynamic plot. """
        PlotBackTest._set_figure(self, fig, ax, size)
        plt.ion()
        return self


    def plot(self, y, x=None, names=None, col='Blues', lw=1., unit='raw', **kwargs):
        """ Plot performances

        Parameters
        ----------
        :x: np.ndarray[ndim=2] with shape=(T, 1)
            x-axis, can be series of int or dates or string.
        :y: np.ndarray[np.float64, ndim=2] with shape=(T, N)
            Returns or indexes.
        
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


    def _set_name(self, name, y, unit='raw', **kwargs):
        if unit.lower() == 'raw':
            return '{}: {:.2f}'.format(name, y[-1])
        elif unit.lower() == 'perf':
            return '{}: {:.0%}'.format(name, y[-1] / y[0] - 1)
        else:
            raise ValueError