#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-11 10:10:43
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-14 17:15:52

""" Some wrappers functions. """

# Built-in packages
from functools import wraps
from warnings import warn

# Third party packages
import numpy as np

# Local packages


def _check_dtype(X, dtype):
    if dtype is None:
        dtype = X.dtype

    if X.dtype != np.float64:
        X = X.astype(np.float64)

    return X, dtype


def wrap_dtype(func):
    """ Check the dtype of the `X` array.

    Convert dtype of X to np.float64 before to pass to cython function and
    convert to specified dtype at the end.

    """
    @wraps(func)
    def check_dtype(X, *args, dtype=None, **kwargs):
        X, dtype = _check_dtype(X, dtype)

        if dtype != np.float64:
            return func(X, *args, dtype=dtype, **kwargs).astype(dtype)

        return func(X, *args, dtype=dtype, **kwargs)

    return check_dtype


def wrap_axis(func):
    """ Check if computation on `axis` of `X` array is available. """
    @wraps(func)
    def check_axis(X, *args, axis=0, **kwargs):
        shape = X.shape

        if len(X.shape) <= axis:

            raise np.AxisError(axis, len(X.shape))

        elif axis == 1 and len(shape) == 2:

            return func(X.T, *args, axis=0, **kwargs).T

        return func(X, *args, axis=axis, **kwargs)

    return check_axis


def wrap_lags(func):
    """ Check the lag of the `X` array. """
    @wraps(func)
    def check_lags(X, k, *args, axis=0, **kwargs):
        if k <= 0:
            raise ValueError('{} lag is not available value.'.format(k))

        if X.shape[axis] < k:
            k = X.shape[axis]
            warn('{} lags is greater than size {} of series on axis {}'.format(
                k, X.shape[axis], axis
            ))

        return func(X, k, *args, axis=axis, **kwargs)

    return check_lags


class WrapperArray:
    """ Object to wrap numpy arrays.

    This object mix several wrapper functions.

    Parameters
    ----------
    *args : {'dtype', 'axis', 'lags'}

    """

    handler = {
        'dtype': wrap_dtype,
        'axis': wrap_axis,
        'lags': wrap_lags,
    }

    def __init__(self, *args):
        """ Initialize wrapper functions. """
        self.wrappers = {key: self.handler[key] for key in args}

    def __call__(self, func):
        """ Wrap `func`.

        Parameters
        ----------
        func : function
            Function to wrap.

        Returns
        -------
        function
            Wrapped function.

        """
        @wraps(func)
        def wrap(X, *args, **kwargs):
            wrap_func = None
            for k, w in self.wrappers.items():
                if wrap_func is None:
                    wrap_func = w(func)

                else:
                    wrap_func = w(wrap_func)

            return wrap_func(X, *args, **kwargs)

        return wrap
