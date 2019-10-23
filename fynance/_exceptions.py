#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-10-23 17:01:36
# @Last modified by: ArthurBernard
# @Last modified time: 2019-10-23 17:17:32

""" Define some various richly-typed exceptions. """

# Built-in packages

# Third party packages

# Local packages


class ArraySizeError(ValueError, IndexError):
    """ Size of the array was invalid. """

    def __init__(self, size, axis=None, min_size=None, msg_prefix=None):
        """ Initialize the array size error. """
        msg = 'array of size {}'.format(size)

        if axis is not None:
            msg += ' in axis {}'.format(axis)

        msg += ' is not allowed'

        if min_size is not None:
            msg += ', minimum size is {}'.format(min_size)

        if msg_prefix is not None:
            msg = '{}: {}'.format(msg_prefix, msg)

        super(ArraySizeError, self).__init__(msg)
