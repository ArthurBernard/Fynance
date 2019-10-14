#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-11 10:10:43
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-14 14:55:19

""" Some wrappers functions. """

# Built-in packages
from functools import wraps

# Third party packages
import numpy as np

# Local packages


def wrap_dtype(func):
    """ Check the dtype of the `X` array.

    Convert dtype of X to np.float64 before to pass to cython function and
    convert to specified dtype at the end.

    """
    @wraps(func)
    def check_dtype(X, *args, dtype=None, **kwargs):
        if dtype is None:
            dtype = X.dtype

        if X.dtype != np.float64:
            X = X.astype(np.float64)

        if dtype != np.float64:
            return func(X, *args, dtype=dtype, **kwargs).astype(dtype)

        return func(X, *args, dtype=dtype, **kwargs)

    return check_dtype


def wrap_axis(func):
    """ Check the axis of the `X` array. """
    @wraps(func)
    def check_axis(X, *args, axis=0, **kwargs):
        shape = X.shape

        if len(X.shape) >= axis:

            raise np.AxisError(axis, len(X.shape))

        elif axis == 1 and len(shape) == 2:

            return func(X.T, *args, axis=0, **kwargs).T

        return func(X, *args, axis=axis, **kwargs)

    return check_axis


