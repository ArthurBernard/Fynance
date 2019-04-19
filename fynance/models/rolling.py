#!/usr/bin/env python3
# coding: utf-8

# Built-in packages

# External packages
import numpy as np

# Internal packages
from fynance.models.xgb import XGBData


__all__ = ['RollingBasis']


class RollingBasis:
    """ Basis object to roll a model over a time axis with a train period
    from `t - n` to `t` and a testing period from `t` to `t + s`.

    Attributes
    ----------
    n, s : int
        Respectively size of training and testing period

    """
    # TODO : other methods

    def __init__(self, X, y):
        """ Initialize shape of target """
        self.T = X.shape[0]
        self.y_shape = y.shape

    def __call__(self, train_period, test_period, start=0, end=None,
                 roll_period=None):
        """ Callable method to set terget features data, and model.

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

        Returns
        -------
        RollingBasis
            The rolling basis model.

        """
        # Set size of subperiods
        self.n = train_period
        self.s = test_period
        self.r = test_period if roll_period is None else roll_period

        # Set boundary of period
        self.t = max(self.n - self.r, start)
        self.T = self.T if end is None else min(self.T, end)

        return self

    def __iter__(self):
        """ Set iterative method """
        self.y_eval = np.zeros([self.y_shape])
        self.y_test = np.zeros([self.y_shape])

        return self

    def __next__(self):
        """ Incrementing method """
        # TODO : to finish
        self.t += self.r

        if self.t + self.s > self.T:

            raise StopIteration

        train_slice = slice(self.t - self.s, self.t)
        test_slice = slice(self.t, self.t + self.s)

        self.y_eval[self.t - self.s: self.t] = self.sub_train()
        self.y_test[self.t: self.t + self.s] = self.sub_predict()

        return self


class RollingXGB(RollingBasis):
    """ Rolling version of eXtrem Gradient Boosting model. Model will roll
    train and test periods over a time axis, at time `t` the training period
    is from `t - n` to `t` and the testing period from `t` to `t + s`.

    Attributes
    ----------
    n, s : int
        Respectively size of training and testing period

    """
    # TODO : to finish

    def __init__(self, X, y, **kwargs):
        """ Setting data to XGBoot model.

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

    def sub_train(self):
        pass

"""
Examples
--------

>>> roll_xgb = RollingXGB(X, y)
>>> for pred_eval, pred_test in roll_xgb(256, 64):
>>>     
"""