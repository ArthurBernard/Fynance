#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-21 06:21:39
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-25 21:00:34
# cython: language_level=3, wraparound=False, boundscheck=False

""" Cython basis object to browse data. """

# Built-in packages

# Third party packages
from cython cimport view
import numpy as np
cimport numpy as np

# Local packages

__all__ = ["_BrowserCy", "_BrowserIndexCy", "_BrowserYCy", "_BrowserXCy"]


cdef class _BrowserCy:
    cdef int start, end, step, i

    def __init__(self, start, end, step):
        self.end = end
        self.step = step
        self.start = start

    def __iter__(self):
        self.i = self.start

        return self

    def __next__(self):
        self.i += self.step
        if self.i > self.end:

            raise StopIteration

        return slice(self.i - self.step, self.i)


cdef class _BrowserIndexCy(_BrowserCy):

    def __init__(self, int [:] idx, int step, int start):
        self.end = idx.size
        self.step = step
        self.start = start
        self.idx = idx

    def __next__(self):
        super(_BrowserIndexCy, self).__next__()

        return self.idx[self.i - self.step: self.step]


cdef class _BrowserYCy:
    """ Browse 1 dimensions data. """

    def __init__(self, double [:] Y, int step, int start):
        self.end = Y.size
        self.step = step
        self.start = start
        self.Y = Y

    def __next__(self):
        super(_BrowserYCy, self).__next__()

        return self.Y[self.i - self.step: self.step]


cdef class _BrowserXCy(_BrowserCy):
    """ Browse 2 dimensions data. """
    cdef double [:, :] X

    def __init__(self, X, step, start):
        super(_BrowserXCy, self).__init__(start, X.shape[0], step)
        # _BrowserCy.__init__(self, start, X.shape[0], step)
        # self.end = X.shape[0]
        # self.step = step
        # self.start = start
        self.X = X

    def __iter__(self):
        return self

    def __next__(self):
        _BrowserCy.__next__(self)

        return self.X[self.i - self.step: self.step]


cdef class Shrubbery:
    cdef int width, height

    def __init__(self, w, h):
        self.width = w
        self.height = h

    def describe(self):
        print("This shrubbery is", self.width,
              "by", self.height, "cubits.")


cdef class _RollingBasisCy:
    cdef int T, n, s, r, b, e, t0, n_iter
    cdef double [:] y_eval

    def __init__(self, double [:, :] X, double [:, :] y):
        self.y_eval = view.array(shape=(X.shape[0],), itemsize=sizeof(double), format='d')
