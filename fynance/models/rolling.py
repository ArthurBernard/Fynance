#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-04-23 19:15:17
# @Last modified by: ArthurBernard
# @Last modified time: 2019-06-24 17:02:30

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

# Local packages
from fynance.models.xgb import XGBData
from fynance.models.neural_network import MultiLayerPerceptron


__all__ = ['RollingBasis', 'RollMultiLayerPerceptron']


class RollingBasis:
    """ Base object to roll a model.

    Rolling over a time axis with a train period from `t - n` to `t` and a
    testing period from `t` to `t + s`.

    Attributes
    ----------
    n, s, r : int
        Respectively size of training, testing and rolling period.
    b, e, T : int
        Respectively batch size, number of epochs and size of entire dataset.
    t : int
        The current time period.
    y_train, y_eval : np.ndarray[ndim=1, dtype=np.float64]
        Respectively training and evaluating predictions.

    """

    # TODO : other methods
    def __init__(self, X, y):
        """ Initialize shape of target. """
        self.T = X.shape[0]
        self.y_shape = y.shape

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
            Number of epochs, default is 1.

        Returns
        -------
        RollingBasis
            The rolling basis model.

        """
        # Set size of subperiods
        self.n = train_period
        self.s = test_period
        self.r = test_period if roll_period is None else roll_period
        self.b = batch_size
        self.e = epochs

        # Set boundary of period
        self.t = max(self.n - self.r, start)
        self.T = self.T if end is None else min(self.T, end)

        return self

    def __iter__(self):
        """ Set iterative method. """
        self.y_eval = np.zeros(self.y_shape)
        self.y_test = np.zeros(self.y_shape)

        return self

    def __next__(self):
        """ Incrementing method. """
        # TODO : to finish
        # Time forward incrementation
        self.t += self.r

        if self.t + self.s > self.T:

            raise StopIteration

        # Run epochs
        for epoch in range(self.e):
            # Run batchs
            for t in range(self.t - self.n, self.t, self.b):
                # Set new train periods
                s = min(t + self.b, self.t)
                train_slice = slice(t, s)
                # Train model
                self._train(X=self.X[train_slice], y=self.y[train_slice])

        # Set new test periods
        test_slice = slice(self.t, self.t + self.s)
        # Predict on training and testing period
        self.y_eval[train_slice] = self.sub_predict(self.X[train_slice])
        self.y_test[test_slice] = self.sub_predict(self.X[test_slice])

        return self

    def run(self):
        """ Running neural network model """
        # TODO : get stats, loss, etc.
        # TODO : plot loss, perf, etc.
        for _ in self:
            pass

        return self


class RollingXGB(RollingBasis):
    """ Rolling version of eXtrem Gradient Boosting model.

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
        RollingBasis.__init__(self, X, y)
        self.data = XGBData(X, label=y, **kwargs)
        self.bst = None

    def _train(self):
        # self.bst = xgb.train(params, )
        pass


class RollMultiLayerPerceptron(MultiLayerPerceptron, RollingBasis):
    """ Rolling version of the vanilla neural network model.

    TODO:
    - fix train and predict methods
    - finish docstring
    - finish methods

    """

    def __init__(self, X, y, layers=[], activation=None, drop=None):
        RollingBasis.__init__(self, X, y)
        MultiLayerPerceptron.__init__(self, X, y, layers=layers,
                                      activation=activation, drop=drop)

    def set_roll_period(self, train_period, test_period, start=0, end=None,
                        roll_period=None, eval_period=None, batch_size=64):
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

        Returns
        -------
        RollingBasis
            The rolling basis model.

        """
        return RollingBasis.__call__(self, train_period, test_period,
                                     start=start, end=end,
                                     roll_period=roll_period,
                                     eval_period=eval_period,
                                     batch_size=batch_size)

    def _train(self, X, y):
        return self.train_on(X, y)

    def sub_predict(self, X):
        return self.predict(X)
