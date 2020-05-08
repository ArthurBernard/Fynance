#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-04-23 19:15:17
# @Last modified by: ArthurBernard
# @Last modified time: 2020-05-07 09:58:12

""" Basis of rolling models.

Examples
--------
# >>> roll_xgb = RollingXGB(X, y)
# >>> for pred_eval, pred_test in roll_xgb(256, 64):
# >>>     plot(pred_eval, pred_test)

"""

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import torch

# Local packages
# from fynance.models.xgb import XGBData
from fynance.models.neural_network import MultiLayerPerceptron
from fynance.backtest.dynamic_plot_backtest import DynaPlotBackTest

# Set plot style
plt.style.use('seaborn')


__all__ = ['_RollingBasis', 'RollMultiLayerPerceptron']


class _RollingBasis:
    """ Base object to roll a neural network model.

    Rolling over a time axis with a train period from `t - n` to `t` and a
    testing period from `t` to `t + s`.

    Parameters
    ----------
    X, y : array_like
        Respectively input and output data.
    f : callable, optional
        Function to transform target, e.g. ``torch.sign`` function.
    index : array_like, optional
        Time index of data.

    Methods
    -------
    __call__
    run

    Attributes
    ----------
    n, s, r : int
        Respectively size of training, testing and rolling period.
    b, e, T : int
        Respectively batch size, number of epochs and size of entire dataset.
    t : int
        The current time period.
    y_eval, y_test : np.ndarray[ndim=1 or 2, dtype=np.float64]
        Respectively evaluating (or training) and testing predictions.

    """

    # TODO : other methods
    def __init__(self, X, y, f=None, index=None):
        """ Initialize shape of target. """
        self.T = X.shape[0]
        self.y_shape = y.shape

        if f is None:
            self.f = lambda x: x

        else:
            self.f = f

        if index is None:
            self.idx = np.arange(self.T)

        else:
            self.idx = index

    # TODO : fix callable method to overwritten problem with torch.nn.Module
    def __call__(self, train_period, test_period, start=0, end=None,
                 roll_period=None, eval_period=None, batch_size=64, epochs=1):
        """ Callable method to set target features data, and model.

        Parameters
        ----------
        train_period, test_period : int
            Size of respectively training and testing sub-periods.
        start : int, optional
            Starting observation, default is first observation.
        end : int, optional
            Ending observation, default is last observation.
        roll_period : int, optional
            Size of the rolling period, default is the same size of the
            testing sub-period.
        eval_period : int, optional
            Size of the evaluating period, default is the same size of the
            testing sub-period if training sub-period is large enough.
        batch_size : int, optional
            Size of a training batch, default is 64.
        epochs : int, optional
            Number of epochs on the same subperiod, default is 1.

        Returns
        -------
        _RollingBasis
            The rolling basis model.

        """
        # Set size of subperiods
        self.n = train_period
        self.s = test_period
        self.r = test_period if roll_period is None else roll_period
        self.b = batch_size
        self.e = epochs

        # Set boundary of period
        self.T = self.T if end is None else min(self.T, end)
        self.t0 = max(self.n - self.r, min(start, self.T - self.n - self.s))

        return self

    def __iter__(self):
        """ Set iterative method. """
        self.y_eval = np.zeros(self.y_shape)
        self.y_test = np.zeros(self.y_shape)
        self.loss_train = []
        self.loss_eval = []
        self.loss_test = []
        # self.t_idx = np.arange(self.T)
        self._e = self.e
        self.t = self.t0

        return self

    def __next__(self):
        """ Incrementing method. """
        # TODO : to finish
        self._e += 1
        if self._e > self.e:
            self._e = 1
            # Time forward incrementation
            self.t += self.r

            if self.t + self.s > self.T:

                raise StopIteration

            self.t_idx = np.arange(self.t - self.n, self.t)
        # TODO : Set training part in an other method
        # Run epochs
        # for epoch in range(self.e):
        loss_epoch = 0.
        # Shuffle time indexes
        np.random.shuffle(self.t_idx)
        # Run batchs
        # for t in range(self.t - self.n, self.t, self.b):
        for t in range(0, self.n, self.b):
            # Set new train periods
            # s = min(t + self.b, self.t)
            s = min(t + self.b, self.n)
            train_slice = self.t_idx[t: s]  # slice(t, s)
            # Train model
            try:
                lo = self._train(
                    X=self.X[train_slice],
                    y=self.f(self.y[train_slice]),
                )
            except Exception as e:
                print(train_slice)
                print(self.X[train_slice])
                print(self.f(self.y[train_slice]))
                raise e
            loss_epoch += lo.item()

        self.loss_train += [loss_epoch / s]

        # Set eval and test periods
        return slice(self.t - self.r, self.t), slice(self.t, self.t + self.s)

    def run(self, backtest_plot=True, backtest_kpi=True, figsize=(9, 6)):
        """ Run neural network model.

        Parameters
        ----------
        backtest_plot : bool, optional
            If True, display plot of backtest performances.
        backtest_kpi : bool, optional
            If True, display kpi of backtest performances.

        """
        y = self.y.numpy()
        y_perf = np.exp(np.cumsum(y, axis=0))
        perf_eval = 100. * np.ones(y.shape)
        perf_test = 100. * np.ones(y.shape)

        # Set dynamic plot object
        f, (ax_1, ax_2) = plt.subplots(2, 1, figsize=figsize)
        plt.ion()
        ax_loss = DynaPlotBackTest(
            f, ax_1, title='Model loss', ylabel='Loss', xlabel='Epochs',
            yscale='log', tick_params={'axis': 'x', 'labelsize': 10}
        )
        ax_perf = DynaPlotBackTest(
            f, ax_2, title='Model perf.', ylabel='Perf.',
            xlabel='Date', yscale='log',
            tick_params={'axis': 'x', 'rotation': 30, 'labelsize': 10}
        )

        # TODO : get stats, loss, etc.
        # TODO : plot loss, perf, etc.
        for eval_slice, test_slice in self:
            # Predict on training and testing period
            self.y_eval[eval_slice] = self.sub_predict(self.X[eval_slice])
            self.y_test[test_slice] = self.sub_predict(self.X[test_slice])
            # Compute losses
            self.loss_eval += [self._get_loss(self.y_eval[eval_slice],
                                              y[eval_slice])]
            self.loss_test += [self._get_loss(self.y_test[test_slice],
                                              y[test_slice])]

            if backtest_kpi:
                self._display_kpi()

            if backtest_plot:
                ax_loss.ax.clear()
                # Plot loss
                ax_loss.plot(np.array(self.loss_test), names='Test',
                             col='BuGn', lw=2.)
                ax_loss.plot(np.array(self.loss_train), names='Train',
                             col='RdPu', lw=1.)
                ax_loss.plot(
                    np.array(self.loss_eval), names='Eval', col='YlOrBr',
                    loc='upper right', ncol=2, fontsize=10, handlelength=0.8,
                    columnspacing=0.5, frameon=True, lw=1.,
                )
                ax_loss.ax.plot(
                    np.arange(0, len(self.loss_test), self.e),
                    np.array([self.loss_test]).T[::self.e],
                    'r.', lw=3,
                )
                # ax_loss.set_axes()
                ax_loss.ax.set_yscale('log')
                ax_loss.ax.set_ylabel('Loss')
                ax_loss.ax.set_xlabel('Epochs')
                ax_loss.ax.tick_params(axis='x', labelsize=10)

                if self._e == self.e:
                    # Set performances of training period
                    returns = np.sign(self.y_eval[eval_slice]) * y[eval_slice]
                    cumret = np.exp(np.cumsum(returns, axis=0))
                    perf_eval[eval_slice] = perf_eval[self.t - self.r - 1] * cumret

                    # Set performances of estimated period
                    returns = np.sign(self.y_test[test_slice]) * y[test_slice]
                    cumret = np.exp(np.cumsum(returns, axis=0))
                    perf_test[test_slice] = perf_test[self.t - 1] * cumret

                    ax_perf.ax.clear()
                    # Plot perf of the test set
                    ax_perf.plot(
                        perf_test[self.t0: self.t + self.s],
                        x=self.idx[self.t0: self.t + self.s],
                        names='Test set', col='GnBu', lw=1.7, unit='perf',
                    )
                    # Plot perf of the eval set
                    ax_perf.plot(
                        perf_eval[self.t0: self.t],
                        x=self.idx[self.t0: self.t],
                        names='Eval set', col='OrRd', lw=1.2, unit='perf'
                    )
                    # Plot perf of the underlying
                    ax_perf.plot(
                        100 * y_perf[self.t0: self.t] / y_perf[self.t0],
                        x=self.idx[self.t0: self.t],
                        names='Underlying', col='RdPu', lw=1.2, unit='perf'
                    )
                    # ax_perf.set_axes()
                    ax_perf.ax.set_yscale('log')
                    ax_perf.ax.set_ylabel('Perf.')
                    ax_perf.ax.set_xlabel('Date')
                    ax_perf.ax.tick_params(axis='x', rotation=30, labelsize=10)
                    ax_perf.ax.legend(loc='upper left', fontsize=10, frameon=True,
                                      handlelength=0.8, ncol=2, columnspacing=0.5)

                f.canvas.draw()
                # plt.draw()

        return self

    def _get_loss(self, input, target):
        # input, target : np.array
        # Compute loss function
        lo = self.criterion(torch.from_numpy(input), torch.from_numpy(target))

        return lo.item()

    def _display_kpi(self):
        # Display %
        pct = self.t - self.n - self.s
        pct = pct / (self.T - self.n - self.T % self.s)
        txt = '{:5.2%} is done | '.format(pct)
        txt += 'Eval loss is {:5.2} | '.format(self.loss_eval[-1])
        txt += 'Test loss is {:5.2} | '.format(self.loss_test[-1])
        print(txt, end='\r')


def get_perf(signal, underlying, v0=100):
    return v0 * np.exp(np.cumsum(signal * underlying, axis=0))


class RollingXGB(_RollingBasis):
    """ Rolling version of eXtrem Gradient Boosting model. NOT YET IMPLEMETED.

    Model will roll train and test periods over a time axis, at time `t` the
    training period is from `t - n` to `t` and the testing period from `t` to
    `t + s`.

    Attributes
    ----------
    n, s : int
        Respectively size of training and testing period.

    """

    # TODO : to finish
    def __init__(self, X, y, **kwargs):
        """ Set data to XGBoot model.

        Parameters
        ----------
        X, y : np.ndarray[ndim=2, dtype=np.float64]
            Respectively features with shape `(T, N)` and target with shape
            `(T, 1)` of the model.
        kwargs : dict, optional
            Parameters of DMatrix object, cf XGBoost documentation [1]_.

        References
        ----------
        .. [1] https://xgboost.readthedocs.io/en/latest/python/python_api.html

        """
        _RollingBasis.__init__(self, X, y)
        # self.data = XGBData(X, label=y, **kwargs)
        self.bst = None

    def _train(self):
        # self.bst = xgb.train(params, )
        pass


class RollMultiLayerPerceptron(MultiLayerPerceptron, _RollingBasis):
    """ Rolling version of the vanilla neural network model.

    Methods
    -------
    run
    set_roll_period
    sub_predict
    save

    TODO:
    - fix train and predict methods
    - finish docstring
    - finish methods

    """

    def __init__(self, X, y, layers=[], activation=None, drop=None, bias=True,
                 x_type=None, y_type=None, activation_kwargs={}, **kwargs):
        """ Initialize rolling multi-layer perceptron model. """
        _RollingBasis.__init__(self, X, y, **kwargs)
        MultiLayerPerceptron.__init__(self, X, y, layers=layers, bias=bias,
                                      activation=activation, drop=drop,
                                      x_type=x_type, y_type=y_type,
                                      activation_kwargs=activation_kwargs)

    def set_roll_period(self, train_period, test_period, start=0, end=None,
                        roll_period=None, eval_period=None, batch_size=64,
                        epochs=1):
        """ Callable method to set target features data, and model.

        Parameters
        ----------
        train_period, test_period : int
            Size of respectively training and testing sub-periods.
        start : int, optional
            Starting observation, default is first observation.
        end : int, optional
            Ending observation, default is last observation.
        roll_period : int, optional
            Size of the rolling period, default is the same size of the
            testing sub-period.
        eval_period : int, optional
            Size of the evaluating period, default is the same size of the
            testing sub-period if training sub-period is large enough.
        batch_size : int, optional
            Size of a training batch, default is 64.
        epochs : int, optional
            Number of epochs, default is 1.

        Returns
        -------
        _RollingBasis
            The rolling basis model.

        """
        return _RollingBasis.__call__(
            self, train_period=train_period, test_period=test_period,
            start=start, end=end, roll_period=roll_period,
            eval_period=eval_period, batch_size=batch_size, epochs=epochs
        )

    def _train(self, X, y):
        return self.train_on(X=X, y=y)

    def sub_predict(self, X):
        """ Predict. """
        return self.predict(X=X).numpy()

    def save(self, path):
        """ Save the trained neural network model.

        Parameters
        ----------
        path : str
            Path to save the model.

        """
        pass
