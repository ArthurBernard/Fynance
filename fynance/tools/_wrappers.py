#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-11 10:10:43
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-12 08:59:46

""" Some wrappers functions. """

# Built-in packages
from functools import wraps

# Third party packages
import numpy as np

# Local packages


def wrap_dtype(func):
    """ Check the dtype of the x array.

    Convert dtype of x to np.float64 before to pass to cython function and
    convert to specified dtype at the end.

    """
    @wraps(func)
    def check_dtype(x, *args, dtype=None, **kwargs):
        if dtype is None:
            dtype = x.dtype

        if x.dtype != np.float64:
            x = x.astype(np.float64)

        if dtype != np.float64:
            return func(x, *args, dtype=dtype, **kwargs).astype(dtype)

        return func(x, *args, dtype=dtype, **kwargs)

    return check_dtype
