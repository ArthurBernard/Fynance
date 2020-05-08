#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-05 19:17:04
# @Last modified by: ArthurBernard
# @Last modified time: 2020-05-08 20:15:27

""" Module with some function plot backtest. """

# Built-in packages

# External packages
from matplotlib import pyplot as plt

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

    def set_axes(self, **kwargs):
        """ Set axes with initial parameters.

        Parameters
        ----------
        **kwargs : keyword arguments, optioanl
            Axes configuration, cf matplotlib documentation [1]_. By default,
            parameters specified in __init__ method are used.

        References
        ----------
        .. [1] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

        """
        ax_params = self.ax_params.copy()
        ax_params.update(kwargs)
        self._set_axes(**ax_params)

    def _set_axes(self, yscale='linear', xscale='linear', ylabel='',
                  xlabel='', title='', tick_params={}):
        """ Set axes parameters. """
        self.ax.set_yscale(yscale)
        # FIXME : the below line avoid to display date on x-axis
        # self.ax.set_xscale(xscale)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel, x=0.9)
        self.ax.set_title(title)
        self.ax.tick_params(**tick_params)

        return self

    def clear(self):
        """ Clear axes. """
        self.ax.clear()


class BacktestNeuralNet:

    def __init__(self, figsize=(9, 6)):
        # Set dynamic plot object
        self.f, (ax_1, ax_2) = plt.subplots(2, 1, figsize=figsize)
        plt.ion()
        self.dp_loss = DynaPlotBackTest(
            self.f, ax_1, title='Model loss', ylabel='Loss', xlabel='Epochs',
            yscale='log', tick_params={'axis': 'x', 'labelsize': 10}
        )
        self.dp_perf = DynaPlotBackTest(
            self.f, ax_2, title='Model perf.', ylabel='Perf.',
            xlabel='Date', yscale='log',
            tick_params={'axis': 'x', 'rotation': 30, 'labelsize': 10}
        )

    def plot_loss(self, test, eval, train=None):
        """ Plot loss function values for test and evaluate set. """
        self.dp_loss.clear()
        # Plot loss
        self.dp_loss.plot(test, names='Test', col='BuGn', lw=2.)
        if train is not None:
            self.dp_loss.plot(train, names='Train', col='RdPu', lw=1.)

        self.dp_loss.plot(eval, names='Eval', col='YlOrBr', loc='upper right',
                          ncol=2, fontsize=10, handlelength=0.8,
                          columnspacing=0.5, frameon=True, lw=1.)
        self.dp_loss.set_axes()

    def plot_perf(self, test, eval, underlying=None, index=None):
        """ Plot performance values for test and eval set. """
        self.dp_perf.clear()
        if index is not None:
            idx_test = index[-test.shape[0]:]
            idx_eval = index[: eval.shape[0]]

        else:
            idx_test = idx_eval = None

        # Plot perf of the test set
        self.dp_perf.plot(test, x=idx_test, names='Test set', col='GnBu',
                          lw=1.7, unit='perf')
        # Plot perf of the eval set
        self.dp_perf.plot(eval, x=idx_eval, names='Eval set', col='OrRd',
                          lw=1.2, unit='perf')
        # Plot perf of the underlying
        if underlying is not None:
            self.dp_perf.plot(underlying, x=idx_eval, names='Underlying',
                              col='RdPu', lw=1.2, unit='perf')
        self.dp_perf.set_axes()
        self.dp_perf.ax.legend(loc='upper left', fontsize=10, frameon=True,
                               handlelength=0.8, ncol=2, columnspacing=0.5)
