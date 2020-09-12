#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-12 09:15:33
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-12 10:22:09

""" Script with basic models. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages

__all__ = []


def _prob_sign(X):
    X_ = np.ones(X.shape)
    X_[X < 0.5] = -1.

    return X_


class SignalModel:
    r""" A basic model designed to estimate signals.

    Parameters
    ----------
    pred : str {'sign', 'proba'}, optional
        - 'sign' (default) means the prediction is positive or negative and
          keep only the sign to compute the signal time series, i.e
          :math:`signal = 1 \text{ if y > 0} -1 \text{ otherwise}`.
        - 'prob' means the prediction is the probability to get a positive
          signal, i.e
          :math:`signal = 1 \text{ if y > 0.5} -1 \text{ otherwise}`.

    """

    handle_func_signal = {
        "sign": np.sign,
        "prob": _prob_sign,
    }

    def __init__(self, pred='sign'):
        """ Initialize the signal model object. """
        self.pred = pred
        self._get_signal = self.handle_func_signal[pred]

    def get_signal(self):
        """ Compute signal of prediction. """
        return self._get_signal(self.y_pred)


class MagnitudeModel:
    """ A basic model designed to estimate magnitudes. """

    pass


if __name__ == "__main__":
    pass
