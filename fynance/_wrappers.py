#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-11 10:10:43
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-24 19:11:46

""" Some wrappers functions. """

# Built-in packages
from functools import wraps
from warnings import warn

# Third party packages
import numpy as np

# Local packages
from fynance._exceptions import ArraySizeError


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
    def check_axis(X, *args, axis=0, min_size=0, **kwargs):
        shape = X.shape
        if shape[axis] < min_size:

            raise ArraySizeError(shape[axis], axis=axis, min_size=min_size)

        elif len(X.shape) <= axis:

            raise np.AxisError(axis, len(X.shape))

        elif axis == 1 and len(shape) == 2:

            return func(X.T, *args, axis=0, **kwargs).T

        return func(X, *args, axis=axis, **kwargs)

    return check_axis


def wrap_lags(func):
    # Not clean, may lead to undesirable behavior
    """ Check the max available lag for `X` array. """
    @wraps(func)
    def check_lags(X, k, *args, axis=0, **kwargs):
        if k <= 0:
            raise ValueError('lag {} must be greater than 0.'.format(k))

        elif X.shape[axis] < k:
            warn('{} lags is out of bounds for axis {} with size {}'.format(
                k, axis, X.shape[axis]
            ))
            k = X.shape[axis]

        return func(X, k, *args, axis=axis, **kwargs)

    return check_lags


def wrap_window(func):
    """ Check if the lagged window `w` is available for `X` array. """
    @wraps(func)
    def check_window(X, w=None, **kwargs):
        if w == 0 or w is None:
            w = X.shape[0]

        elif w < 0:

            raise ValueError('lagged window of size {} is not available, \
                must be positive.'.format(w))

        elif w > X.shape[0]:
            warn('lagged window of size {} is out of bounds with time axis \
                of size {}'.format(w, X.shape[0]))
            w = X.shape[0]

        return func(X, w=int(w), **kwargs)

    return check_window


def wrap_expo(func):
    # Not clean, may lead to undesirable behavior
    """ Check if parameters is allowed by the `kind` of moving avg/std. """
    @wraps(func)
    def check_expo(X, *args, w=None, kind=None, **kwargs):
        if kind == 'e':
            w = 1 - 2 / (1 + w)

        return func(X, *args, w=w, kind=kind, **kwargs)

    return check_expo


def wrap_null(func):
    """ Check if there is any null value and raise an exception. """
    @wraps(func)
    def check_null(X, *args, **kwargs):
        if (X == 0).any():

            raise ValueError('null value in X is not allowed.')

        return func(X, *args, **kwargs)

    return check_null


def wrap_ddof(func):
    """ Check if ddof is not greater then the number of timeframe. """
    @wraps(func)
    def check_ddof(X, *args, ddof=0, **kwargs):
        if ddof >= X.shape[0]:
            msg_prefix = 'with degree of freedom {}'.format(ddof)

            raise ArraySizeError(X.shape[0], msg_prefix=msg_prefix)

            raise ValueError(
                "degree of freedom {} is greater than size {} of X".format(
                    ddof, X.shape[0]
                )
            )

        return func(X, *args, ddof=ddof, **kwargs)

    return check_ddof


class WrapperArray:
    """ Object to wrap function that handle numpy arrays.

    This object mix several wrapper functions.

    Parameters
    ----------
    *args : {'dtype', 'axis', 'lags', 'window', 'null', 'ddof', 'keepdims'}
        Wrapper functions.
    **kwargs
        Keyword arguments to pass to wrapper functions.

    """

    handler = {
        'dtype': wrap_dtype,
        'axis': wrap_axis,
        'lags': wrap_lags,
        'window': wrap_window,
        'null': wrap_null,
        'ddof': wrap_ddof,
    }

    def __init__(self, *args, **kwargs):
        """ Initialize wrapper functions. """
        self.wrappers = {key: self.handler[key] for key in args}
        self.kw = kwargs

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
            kwargs = {**kwargs, **self.kw}

            for k, w in self.wrappers.items():

                if wrap_func is None:
                    wrap_func = w(func)

                else:
                    wrap_func = w(wrap_func)

            return wrap_func(X, *args, **kwargs)

        return wrap
