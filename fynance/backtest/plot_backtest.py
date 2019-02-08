#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Internal packages

# Set plot style
plt.style.use('seaborn')

__all__ = ['PlotBackTest']

# TODO: FINISH DOCSTRING

class PlotBackTest:
    """ Plot backtest
    
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
    def __init__(self, fig=None, ax=None, size=(9, 6), **kwargs):
        """ Init method sets size of training and predicting period, inital 
        value to backtest, a target filter and training parameters.

        Parameters
        ----------
        :fig: matplotlib.figure.Figure
            Figure to display backtest.
        :ax: matplotlib.axes
            Axe(s) to display a part of backtest.

        """
        
        # Set Figure
        self._set_figure(fig, ax, size)
        
        # Set axes
        self._set_axes(**kwargs)
        

    def _set_figure(self, fig, ax, size):
        """ Set figure, axes and parameters for dynamic plot. """
        # Set figure and axes
        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots(1, 1, size)
        else:
            self.fig, self.ax = fig, ax
        return self


    def plot(self, y, x=None, names=None, col='Blues', lw=1., **kwargs):
        """ Plot performances

        Parameters
        ----------
        :y: np.ndarray[np.float64, ndim=2] with shape=(T, N)
            Returns or indexes.
        :x: np.ndarray[ndim=2] with shape=(T, 1)
            x-axis, can be series of int or dates or string.
        
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
            l[i].set_label(self._set_name(names[i]))
        
        # display
        self.ax.legend(**kwargs)
        #self.f.canvas.draw()

        return self


    def _set_name(self, name):
        return '{}'.format(name)


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