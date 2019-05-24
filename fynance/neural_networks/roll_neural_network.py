#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-05-10 12:24:27
# @Last modified by: ArthurBernard
# @Last modified time: 2019-05-24 12:05:01

""" Module with basic rolling neural network. """

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns

# Local packages
from fynance.backtest.dynamic_plot_backtest import DynaPlotBackTest

# Set plot style
plt.style.use('seaborn')


class RollNeuralNet:
    """ Object to train/test a neural network along time axis.

    Rolling Neural Network object allow you to train neural networks
    along training periods (from t - n to t) and predict along testing
    periods (from t to t + s) and roll along this time axis.

    Attributes
    ----------
    y : np.ndarray[np.float32, ndim=2] with shape=(T, 1)
        Target to predict, a good practice is to use log-returns.
    X : np.ndarray[np.float32, ndim=2] with shape=(T, N)
        Features (inputs).
    NN : keras.Model
        Neural network to train and predict.
    y_train : np.ndarray[np.float64, ndim=1]
        Prediction on training set.
    y_estim : np.ndarray[np.float64, ndim=1]
        Prediction on estimating set.

    Methods
    -------
    run(y, X, NN, plot_loss=True, plot_perf=True, x_axis=None)
        Train rolling neural networks along pre-specified training period
        and predict along test period. Display loss and performance if
        specified.
    __call__(y, X, NN, start=0, end=1e8, x_axis=None)
        Callable method to set target and features data, neural network
        object (Keras object is prefered).
    __iter__()
        Train and predict along time axis from day number n to last day
        number T and by step of size s period.
    plot_loss(self, f, ax)
        Plot loss function
    plot_perf(self, f, ax)
        Plot perfomances.

    See Also
    --------
    RollMultiNeuralNet, RollAggrMultiNeuralNet, RollMultiRollNeuralNet

    """

    def __init__(self, train_period=252, estim_period=63, value_init=100,
                 target_filter='sign', params=None):
        """ Initialize method.

        Set size of training and predicting period, inital
        value to backtest, a target filter and training parameters.

        Parameters
        ----------
        train_period : int, optional
            Size of the training period. Default is 252 corresponding at
            one year i.e 252 trading days.
        estim_period : int, optional
            Size of the period to predict (is also the default rolling
            period). Default is 63 corresponding at three months i.e 63
            trading days.
        value_init : int, optional
            Initial value to backtest strategy. Default is 100.
        target_filter : function, str or bool
            Function to filtering target. If True or 'sign' use np.sign()
            function, if False doesn't filtering target. Default is 'sign'.
        params : dict, optional
            Parameters for training periods

        """
        self.n = train_period
        self.s = estim_period
        self.V0 = value_init
        self._set_parameters(params)
        if target_filter or target_filter == 'sign':
            self.f = np.sign
        elif target_filter is None:
            self.f = lambda x: x
        else:
            self.f = target_filter

    def __call__(self, y, X, NN, start=0, end=1e8, x_axis=None):
        """ Callable method.

        Set terget and features data, neural
        network object (Currently Keras object is prefered).

        Parameters
        ----------
        y : np.ndarray[ndim=1, dtype=np.float32]
            Target to predict.
        X : np.ndarray[ndim=2, dtype=np.float32]
            Features data.
        NN : keras.engine.training.Model
            Neural network model.
        start : int, optional
            Starting observation, default is first observation.
        end : int, optional
            Ending observation, default is last observation.
        x_axis : np.ndarray[ndim=1], optional
            X-Axis to use for the backtest (int, date, str, etc.).

        Returns
        -------
        rnn : RollNeuralNet

        """
        # Set target and features
        self.y = y
        self.X = X

        # Set neural network model
        self.NN = NN

        # Set periodicity
        self.t = max(self.n, start)
        self.T = min(y.size, end)
        if x_axis is None:
            self.x_axis = range(self.T)
        else:
            self.x_axis = x_axis

        return self

    def __iter__(self):
        """ Set iterative method. """
        self.y_train = np.zeros([self.T, 1])
        self.y_estim = np.zeros([self.T, 1])
        k = self.params['epochs'] * (self.T - self.t) // self.s
        self.loss_train = np.ones([k, 1])
        self.loss_estim = np.ones([k, 1])
        return self

    def __next__(self):
        """ Increment method. """
        # Incremant time
        self.t += self.s
        t = self.t

        if self.t >= self.T:

            raise StopIteration

        # Splitting
        subtrain_X = self.X[t - self.n: t, :]
        subtrain_y = self.f(self.y[t - self.n: t, :])
        subestim_X = self.X[t: t + self.s, :]
        subestim_y = self.f(self.y[t: t + self.s, :])

        # Training
        self.y_train[t - self.s: t, 0] = self._train(
            y=subtrain_y, X=subtrain_X,
            val_set=(subestim_X, subestim_y)
        )

        # Estimating
        self.y_estim[t: t + self.s, 0] = self.NN.predict(
            subestim_X
        ).flatten()

        return self.y_train[t - self.s: t], self.y_estim[t: t + self.s]

    def _train(self, y, X, val_set=None):
        """ Train method and return prediction on training set. """
        # Set parameters
        k = self.params['epochs'] * ((self.t - self.n) // self.s - 1)
        k_1 = self.params['epochs'] * (self.t - self.n) // self.s

        # Fit loss
        hist = self.NN.fit(
            x=X, y=y, validation_data=val_set, **self.params
        )
        self.loss_train[k: k_1, 0] = hist.history['loss']
        self.loss_estim[k: k_1, 0] = hist.history['val_loss']

        # Return prediction
        return self.NN.predict(
            X[-self.s:], verbose=self.params['verbose']
        ).flatten()

    def _set_parameters(self, params):
        """ Set parameters to fit method of neural network.

        If is `None` set as default parameters: `batch_size=train_period`,
        `epochs=1`, `shuffle=False` and no verbosity.

        Parameters
        ----------
        params : dict
            Parameters for training periods

        Returns
        -------
        rnn : RollNeuralNet

        """
        if params is None:
            self.params = {
                'batch_size': self.n,
                'epochs': 1,
                'shuffle': False,
                'verbose': 0
            }
        else:
            self.params = params

        return self

    def run(self, y, X, NN, plot_loss=True, plot_perf=True, x_axis=None):
        """ Train rolling neural networks along time axis.

        Display loss and performance if specified.

        Parameters
        ----------
        y : np.ndarray[np.float32, ndim=2], with shape (T, 1)
            Time series of target to estimate or predict.
        X : np.ndarray[np.float32, ndim=2], with shape (T, N)
            Several time series of features.
        NN : keras.Model or list of keras.Model
            Neural networks to train and predict.
        plot_loss : bool, optional
            If true dynamic plot of loss function.
        plot_perf : bool, optional
            If true dynamic plot of strategy performance.
        x_axis : list or array, optional
            x-axis to plot (e.g. list of dates).

        Returns
        -------
        rnn : RollNeuralNet

        """
        # Set perf and loss arrays
        self.perf_train = self.V0 * np.ones([y.size, 1])
        self.perf_estim = self.V0 * np.ones([y.size, 1])

        # Set axes and figure
        f, ax_loss, ax_perf = self._set_figure(plot_loss, plot_perf)

        # Start Rolling Neural Network
        for pred_train, pred_estim in self(y, X, NN, x_axis=x_axis):
            t, s = self.t, self.s

            # Set performances of training period
            returns = np.sign(pred_train) * y[t - s: t]
            cum_ret = np.exp(np.cumsum(returns, axis=0))
            self.perf_train[t - s: t] = self.perf_train[t - s - 1] * cum_ret

            # Set performances of estimated period
            returns = np.sign(pred_estim) * y[t: t + s]
            cum_ret = np.exp(np.cumsum(returns, axis=0))
            self.perf_estim[t: t + s] = self.perf_estim[t - 1] * cum_ret

            # Plot loss and perf
            self._dynamic_plot(f, ax_loss=ax_loss, ax_perf=ax_perf)

        return self

    def _dynamic_plot(self, f, ax_loss=None, ax_perf=None):
        """ Dynamic plot. """
        # Plot progress of loss
        if ax_loss is not None:
            self.plot_loss(f, ax_loss)

        # Plot progress of performance
        if ax_perf is not None:
            self.plot_perf(f, ax_perf)

        f.canvas.draw()
        return self

    def plot_loss(self, f, ax):
        """ Plot loss function of train/estim periods of neural network.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to display backtest.
        ax : matplotlib.axes
            Axe(s) to display a part of backtest.

        Returns
        -------
        rnn : RollNeuralNet

        """
        k = self.params['epochs'] * (self.t - self.n) // self.s

        dpbt = DynaPlotBackTest(
            fig=f, ax=ax, title='Model loss', ylabel='Loss',
            xlabel='Epoch', yscale='log',
            tick_params={'axis': 'x', 'labelsize': 10}
        )

        # Set graphs
        dpbt.plot(
            self.loss_estim[: k], names='Estim NN', col='BuGn', lw=2.
        )
        dpbt.plot(
            self.loss_train[: k], names='Train NN', col='YlOrBr', lw=1.5
        )
        ax.legend(loc='upper right', ncol=2, fontsize=10,
                  handlelength=0.8, columnspacing=0.5, frameon=True)

        return self

    def plot_perf(self, f, ax):
        """ Plot performances of train/estim periods of neural network.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to display backtest.
        ax : matplotlib.axes
            Axe(s) to display a part of backtest.

        Returns
        -------
        rnn : RollNeuralNet

        """
        t, t_s = self.t, min(self.t + self.s, self.T)

        dpbt = DynaPlotBackTest(
            fig=f, ax=ax, title='Model performance', ylabel='Perf.',
            xlabel='Date', yscale='log',
            tick_params={'axis': 'x', 'rotation': 30, 'labelsize': 10}
        )

        # Set graphs
        dpbt.plot(
            self.perf_estim[: t_s], x=self.x_axis[: t_s],
            names='Estim NN', col='GnBu', lw=1.7, unit='perf',
        )
        dpbt.plot(
            self.perf_train[: t], x=self.x_axis[: t],
            names='Train NN', col='OrRd', lw=1.2, unit='perf'
        )
        ax.legend(loc='upper left', ncol=2, fontsize=10,
                  handlelength=0.8, columnspacing=0.5, frameon=True)

        return self

    def _set_figure(self, plot_loss, plot_perf):
        """ Set figure, axes and parameters for dynamic plot. """
        # Set figure and axes
        f, ax = plt.subplots(plot_loss + plot_perf, 1, figsize=(9, 6))
        plt.ion()

        # Specify axes
        if plot_loss and not plot_perf:
            ax_loss, ax_perf = ax, None

        elif not plot_loss and plot_perf:
            ax_loss, ax_perf = None, ax

        elif plot_loss and plot_perf:
            ax_loss, ax_perf = ax

        else:
            ax_loss, ax_perf = None, None

        return f, ax_loss, ax_perf
