#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-25 19:08:48
# @Last modified by: ArthurBernard
# @Last modified time: 2020-10-02 21:52:02

""" Basis objects to browse data. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages
from .browsers_cy import *

__all__ = ["BrowserData"]


class BrowserData:
    def __init__(self, X, start=0, end=None, step=1, shuffle=False):
        # super(BrowserData, self).__init__(X, step, start)
        self.end = X[0] if end is None else end

        if not shuffle:
            self.generator = range(start, self.end, step)

        else:
            self.idx = np.arange(start, self.end, step)

        self.X = X
        self.step = step
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
            self.generator = (x for x in self.idx)

        return self.generator.__iter__()

    def __next__(self):
        # return _BrowserXCy.__next__(self)
        i = self.generator.__next__()

        return self.X[i: i + self.step]


class RollingBasis:
    def __init__(self, start, end, step):
        pass


if __name__ == "__main__":
    pass
