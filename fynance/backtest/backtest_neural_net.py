#!/usr/bin/env python3
# coding: utf-8

""" Live multi-panel backtest figure for neural-network training.

Provides :class:`BacktestNeuralNet`, which composes the dynamic
loss / accuracy / performance plots from
:mod:`fynance.backtest.dynamic_plot_backtest` into a single updating figure.
"""

# Third-party packages
from matplotlib import pyplot as plt

# Local packages
from fynance.backtest.dynamic_plot_backtest import (
    DynaPlotAccuracy,
    DynaPlotLoss,
    DynaPlotPerf,
)

__all__ = ['BacktestNeuralNet']


class BacktestNeuralNet:

    def __init__(self, figsize=(9, 6), loss_xlim=None, perf_xlim=None,
                 accu_xlim=None, plot_accuracy=False, plot_loss=True,
                 plot_perf=False, **subplot_kw):
        # Set dynamic plot object
        n_rows = plot_accuracy + plot_loss + plot_perf
        self.f, self.axes = plt.subplots(n_rows, 1, figsize=figsize,
                                         **subplot_kw)

        if n_rows == 1:
            self.axes = [self.axes]

        plt.ion()
        self.accu_is_plot = False
        self.loss_is_plot = False
        self.perf_is_plot = False

        if plot_accuracy:
            self.set_plot_accuracy(self.axes[0], accu_xlim=accu_xlim)

        if plot_loss:
            self.set_plot_loss(self.axes[int(plot_accuracy)],
                               loss_xlim=loss_xlim)

        if plot_perf:
            self.set_plot_perf(self.axes[int(plot_accuracy + plot_loss)],
                               perf_xlim=perf_xlim)

    def set_plot_accuracy(self, ax, accu_xlim=None):
        """ Set plot accuracy object. """
        self.dp_accu = DynaPlotAccuracy(self.f, ax)
        self.dp_accu.ax.grid()
        self.dp_accu.ax.set_autoscaley_on(True)

        if accu_xlim is not None:
            self.dp_accu.ax.set_xlim(*accu_xlim, auto=False)
            print("setup xlim:", accu_xlim)
            self.dp_accu.ax.set_autoscalex_on(False)

        else:
            self.dp_accu.ax.set_autoscalex_on(True)

    def plot_accuracy(self, test, eval, train=None, clear=True):
        """ Plot accuracy scores for test and evaluate set. """
        self.dp_accu.plot(test=test, eval=eval, train=train, clear=clear)
        self.accu_is_plot = True

    def update_accuracy(self, test, eval, train=None):
        """ Plot accuracy scores for test and evaluate set. """
        self.dp_accu.update(test=test, eval=eval, train=train)

    def set_plot_loss(self, ax, loss_xlim=None):
        """ Set plot loss object. """
        self.dp_loss = DynaPlotLoss(self.f, ax)
        self.dp_loss.ax.grid()
        self.dp_loss.ax.set_autoscaley_on(True)

        if loss_xlim is not None:
            self.dp_loss.ax.set_xlim(*loss_xlim, auto=False)
            print("setup xlim:", loss_xlim)
            self.dp_loss.ax.set_autoscalex_on(False)

        else:
            self.dp_loss.ax.set_autoscalex_on(True)

    def plot_loss(self, test, eval, train=None, clear=True):
        """ Plot loss function values for test and evaluate set. """
        self.dp_loss.plot(test=test, eval=eval, train=train, clear=clear)
        self.loss_is_plot = True

    def update_loss(self, test, eval, train=None):
        """ Plot loss function values for test and evaluate set. """
        self.dp_loss.update(test=test, eval=eval, train=train)

    def set_plot_perf(self, ax, perf_xlim=None):
        # set perf plot
        self.dp_perf = DynaPlotPerf(self.f, ax)
        self.dp_perf.ax.grid()
        self.dp_perf.ax.set_autoscaley_on(True)
        if perf_xlim is not None:
            self.dp_perf.ax.set_xlim(*perf_xlim, auto=False)
            print("setup xlim:", perf_xlim)
            self.dp_perf.ax.set_autoscalex_on(False)

        else:
            self.dp_perf.ax.set_autoscalex_on(True)

    def plot_perf(self, test, eval, underlying=None, index=None, clear=True):
        """ Plot performance values for test and eval set. """
        self.dp_perf.plot(test=test, eval=eval, underlying=underlying,
                          index=index, clear=clear)
        self.perf_is_plot = True

    def update_perf(self, test, eval, underlying=None, index=None, clear=True):
        """ Update performance values for test and eval set. """
        self.dp_perf.update(test=test, eval=eval, underlying=underlying,
                            index=index, clear=clear)
