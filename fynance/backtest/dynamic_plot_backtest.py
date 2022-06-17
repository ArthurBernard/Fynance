#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-05 19:17:04
# @Last modified by: ArthurBernard
# @Last modified time: 2022-06-17 12:30:45

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
    plot(y, x=None, label=None, color='Blues', lw=1., **kwargs)
        Plot performances.

    See Also
    --------
    PlotBackTest, display_perf, set_text_stats

    """

    plt.ion()
    test_plot_kw = dict(label='Test set', color='b', lw=2.)
    train_plot_kw = dict(label='Train set', color='r', lw=1.)
    eval_plot_kw = dict(label='Eval set', color='g', lw=1.)
    legend_kw = {
        "loc": "upper right",
        "ncol": 2,
        "fontsize": 10,
        "handlelength": 0.8,
        "columnspacing": 0.5,
        "frameon": True,
    }

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
        # FIXME : not explicitly defined and not saved updated parameters
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


class __BacktestNeuralNet:
    # OLD VERSION => DEPRECIATED

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

    def plot_loss(self, test, eval, train=None, clear=True):
        """ Plot loss function values for test and evaluate set. """
        if clear:
            self.dp_loss.clear()

        # Plot loss
        self.dp_loss.plot(test, label='Test', color='BuGn', lw=2.)
        if train is not None:
            self.dp_loss.plot(train, label='Train', color='RdPu', lw=1.)

        self.dp_loss.plot(eval, label='Eval', color='YlOrBr', loc='upper right',
                          ncol=2, fontsize=10, handlelength=0.8,
                          columnspacing=0.5, frameon=True, lw=1.)
        self.dp_loss.set_axes()

    def plot_perf(self, test, eval, underlying=None, index=None, clear=True):
        """ Plot performance values for test and eval set. """
        if clear:
            self.dp_perf.clear()
        if index is not None:
            idx_test = index[-test.shape[0]:]
            idx_eval = index[: eval.shape[0]]

        else:
            idx_test = idx_eval = None

        # Plot perf of the test set
        self.dp_perf.plot(test, x=idx_test, label='Test set', color='GnBu',
                          lw=1.7, unit='perf')
        # Plot perf of the eval set
        self.dp_perf.plot(eval, x=idx_eval, label='Eval set', color='OrRd',
                          lw=1.2, unit='perf')
        # Plot perf of the underlying
        if underlying is not None:
            self.dp_perf.plot(underlying, x=idx_eval, label='Underlying',
                              color='RdPu', lw=1.2, unit='perf')
        self.dp_perf.set_axes()
        self.dp_perf.ax.legend(loc='upper left', fontsize=10, frameon=True,
                               handlelength=0.8, ncol=2, columnspacing=0.5)


class DynaPlotAccuracy(DynaPlotBackTest):
    """ Plot dynamically the accuracy scores.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure to display backtest.
    ax : matplotlib.axes
        Axe(s) to display a part of backtest.
    ax_kwargs : dict
        Parameters of matplotlib axes containing title, ylabel, xlabel, yscale,
        xscale and ticks_params.

    Methods
    -------
    plot
    set_axe
    update

    See Also
    --------
    DynaPlotPerf, DynaPlotLoss

    """

    ax_kw = {
        "title": "Model Accuracy",
        "ylabel": "Accuracy",
        "xlabel": "Epochs",
        "yscale": "linear",
        # "xscale": "linear",
        "tick_params": {"axis": "x", "labelsize": 10},
    }
    test_plot_kw = {
        "label": "Test set", 
        "color": "b",
        "lw": 1.7,
        "unit": 'perf',
    }
    eval_plot_kw = {
        "label": "Eval set",
        "color": "r",
        "lw": 1.2,
        "unit": "perf",
    }
    train_plot_kw = {
        "label": "Train set",
        "color": "g",
        "lw": 1.2,
        "unit": "perf",
    }

    def __init__(self, fig=None, ax=None, size=(9, 6), **kwargs):
        """ Initialize method.

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
            {'yscale': 'linear', 'xscale': 'linear', 'ylabel': 'Accuracy',
            'xlabel': 'Epoch', 'title': 'Model Accuracy', 'tick_params': {'axis':
            'x', 'labelsize': 10}}

        References
        ----------
        .. [1] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

        """
        self.ax_kw.update(kwargs)
        DynaPlotBackTest.__init__(self, fig=fig, ax=ax, size=size,
                                  **self.ax_kw)

    def plot(self, test, eval, train=None, clear=True):
        """ Plot accuracy scores for test and evaluate set.

        Parameters
        ----------
        test, eval, train : np.ndarray[np.float64, ndim=1]
            Respectively test, eval and train accuracy scores.
        clear : bool, optional
            Clear axes if True (default).

        """
        if clear:
            self.clear()

        # Plot accuracy
        DynaPlotBackTest.plot(self, test, **self.test_plot_kw)
        if train is not None:
            DynaPlotBackTest.plot(self, train, **self.train_plot_kw)

        DynaPlotBackTest.plot(self, eval, **self.eval_plot_kw)
        self.set_axes()
        self.ax.legend(**self.legend_kw)

    def update(self, test, eval, train=None, clear=True):
        """ Update plot accuracy scores for test and evaluate set.

        Parameters
        ----------
        test, eval, train : np.ndarray[np.float64, ndim=1]
            Respectively test, eval and train accuracy scores.

        """
        # Plot accuracy
        DynaPlotBackTest.update(self, test, label=self.test_plot_kw['label'])
        if train is not None:
            DynaPlotBackTest.update(self, train,
                                    label=self.train_plot_kw['label'])

        DynaPlotBackTest.update(self, eval, label=self.eval_plot_kw['label'])

        # rescale
        self.ax.relim()
        self.ax.autoscale_view()


class DynaPlotLoss(DynaPlotBackTest):
    """ Plot dynamically the loss scores.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure to display backtest.
    ax : matplotlib.axes
        Axe(s) to display a part of backtest.
    ax_kwargs : dict
        Parameters of matplotlib axes containing title, ylabel, xlabel, yscale,
        xscale and ticks_params.

    Methods
    -------
    plot
    set_axe
    update

    See Also
    --------
    DynaPlotPerf, DynaPlotAccuracy

    """

    ax_kw = {
        "title": "Model Loss",
        "ylabel": "Loss",
        "xlabel": "Epochs",
        "yscale": "log",
        # "xscale": "linear",
        "tick_params": {"axis": "x", "labelsize": 10},
    }

    def __init__(self, fig=None, ax=None, size=(9, 6), **kwargs):
        """ Initialize method.

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
            {'yscale': 'log', 'xscale': 'linear', 'ylabel': 'Loss',
            'xlabel': 'Epoch', 'title': 'Model Loss', 'tick_params': {'axis':
            'x', 'labelsize': 10}}

        References
        ----------
        .. [1] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

        """
        self.ax_kw.update(kwargs)
        DynaPlotBackTest.__init__(self, fig=fig, ax=ax, size=size,
                                  **self.ax_kw)

    def plot(self, test, eval, train=None, clear=True):
        """ Plot loss function values for test and evaluate set.

        Parameters
        ----------
        test, eval, train : np.ndarray[np.float64, ndim=1]
            Respectively test, eval and train loss scores.
        clear : bool, optional
            Clear axes if True (default).

        """
        if clear:
            self.clear()

        # Plot loss
        DynaPlotBackTest.plot(self, test, **self.test_plot_kw)
        if train is not None:
            DynaPlotBackTest.plot(self, train, **self.train_plot_kw)

        DynaPlotBackTest.plot(self, eval, **self.eval_plot_kw)
        self.set_axes()
        self.ax.legend(**self.legend_kw)

    def update(self, test, eval, train=None, clear=True):
        """ Update plot loss function values for test and evaluate set.

        Parameters
        ----------
        test, eval, train : np.ndarray[np.float64, ndim=1]
            Respectively test, eval and train loss scores.

        """
        # Plot loss
        DynaPlotBackTest.update(self, test, label=self.test_plot_kw['label'])
        if train is not None:
            DynaPlotBackTest.update(self, train,
                                    label=self.train_plot_kw['label'])

        DynaPlotBackTest.update(self, eval, label=self.eval_plot_kw['label'])

        # rescale
        self.ax.relim()
        self.ax.autoscale_view()


class DynaPlotPerf(DynaPlotBackTest):
    """ Plot dynamically the performance values.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure to display backtest.
    ax : matplotlib.axes
        Axe(s) to display a part of backtest.
    ax_kwargs : dict
        Parameters of matplotlib axes containing title, ylabel, xlabel, yscale,
        xscale and ticks_params.

    Methods
    -------
    plot
    set_axe
    update

    See Also
    --------
    DynaPlotPerf, DynaPlotAccuracy

    """
        
    ax_kw = {
        "title": "Model Perf",
        "ylabel": "Perf.",
        "xlabel": "Date",
        "yscale": "log",
        # "xscale": "linear",
        "tick_params": {"axis": "x", "rotation": 30, "labelsize": 10},
    }
    test_plot_kw = {
        "label": "Test set", 
        "color": "b",
        "lw": 1.7,
        "unit": 'perf',
    }
    eval_plot_kw = {
        "label": "Eval set",
        "color": "r",
        "lw": 1.2,
        "unit": "perf",
    }
    under_plot_kw = {
        "label": "Underlying",
        "color": "g",
        "lw": 1.2,
        "unit": "perf"
    }
    legend_kw = {
        "loc": "upper left",
        "ncol": 2,
        "fontsize": 10,
        "handlelength": 0.8,
        "columnspacing": 0.5,
        "frameon": True,
    }

    def __init__(self, fig=None, ax=None, size=(9, 6), **kwargs):
        """ Initialize method.

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
            {'yscale': 'log', 'xscale': 'linear', 'ylabel': 'Perf.',
            'xlabel': 'Epoch', 'title': 'Model Perf', 'tick_params': {'axis':
            'x', 'roation': 30, 'labelsize': 10}}

        References
        ----------
        .. [1] https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

        """
        self.ax_kw.update(kwargs)
        DynaPlotBackTest.__init__(self, fig=fig, ax=ax, size=size,
                                  **self.ax_kw)

    def plot(self, test, eval, underlying=None, index=None, clear=True):
        """ Plot performance results of the model for test and evaluate set.

        Parameters
        ----------
        test, eval : np.ndarray[np.float64, ndim=1]
            Respectively test and eval performance results of the model.
        underlying : np.ndarray[np.float64, ndim=1]
            Performance results of the underlying.
        index :

        clear : bool, optional
            Clear axes if True (default).

        """
        if clear:
            self.clear()

        # Set index
        if index is not None:
            idx_test = index[-test.shape[0]:]
            idx_eval = index[: eval.shape[0]]

        else:
            idx_test = idx_eval = None

        # Plot perf
        DynaPlotBackTest.plot(self, test, x=idx_test, **self.test_plot_kw)
        DynaPlotBackTest.plot(self, eval, x=idx_eval, **self.eval_plot_kw)

        # Plot perf of the underlying
        if underlying is not None:
            DynaPlotBackTest.plot(self, underlying, x=idx_eval,
                                  **self.under_plot_kw)

        self.set_axes()
        self.ax.legend(**self.legend_kw)

    def update(self, test, eval, underlying=None, index=None):
        """ Update plot performance results for test and evaluate set.

        Parameters
        ----------
        test, eval : np.ndarray[np.float64, ndim=1]
            Respectively test and eval performance results of the model.
        underlying : np.ndarray[np.float64, ndim=1]
            Performance results of the underlying.
        index :


        """
        # Set index
        if index is not None:
            idx_test = index[-test.shape[0]:]
            idx_eval = index[: eval.shape[0]]

        else:
            idx_test = idx_eval = None

        # Plot perf
        DynaPlotBackTest.update(self, test, label=self.test_plot_kw['label'])
        DynaPlotBackTest.update(self, eval, label=self.eval_plot_kw['label'])
    
        if underlying is not None:
            DynaPlotBackTest.update(self, underlying,
                                    label=self.under_plot_kw['label'])

        # rescale
        self.ax.relim()
        self.ax.autoscale_view()


class _BacktestNeuralNet:
    # TODO : to implement base class

    dyna_plots = {}

    def set_dyna_plot(self, name, klass, ax, **kwargs):
        dyna_plots[name] = klass(self.f,  ax, **kwargs)

    def set_fig_and_axes(self, n_rows, n_cols, figsize):
        self.f, self.axes = plt.subplots(n_rows, n_cols, figsize=figsize)


class BacktestNeuralNet:

    def __init__(self, figsize=(9, 6), loss_xlim=None, perf_xlim=None,
                 accu_xlim=None, plot_accuracy=False, plot_loss=True,
                 plot_perf=False):
        # Set dynamic plot object
        n_rows = plot_accuracy + plot_loss + plot_perf
        self.f, self.axes = plt.subplots(n_rows, 1, figsize=figsize)

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
