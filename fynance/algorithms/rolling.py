#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-09-13 10:27:51
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-20 10:08:14

""" Object to roll algorithms. """

# Built-in packages

# Third party packages

# Local packages

__all__ = ['_RollingMechanism']


class _RollingMechanism:
    """ Rolling mechanism. """

    def __init__(self, index, n, s):
        """ Initialize object. """
        self.idx = index
        self.n = n
        self.s = s

    def __call__(self, t=None, T=None, display=True):
        self.display = display
        # Set initial index
        if t is None:
            self.t = self.n

        else:
            self.t = t

        # Set max index
        if T is None:
            self.T = self.idx.size

        else:
            self.T = T

        return self

    def __iter__(self):
        # Iterative method
        return self

    def __next__(self):
        # Next method
        t = self.t

        # Display %
        if self.display:
            self._display()

        if self.t >= self.T - 1:

            raise StopIteration

        # Update rolling
        self.t += self.s

        # Set indexes
        self.d_n = self.idx[max(t - self.n, 0)]
        self.d_1 = self.idx[t - 1]
        self.d = self.idx[t]
        self.d_s = self.idx[min(t + self.s, self.T - 1)]

        return slice(self.d_n, self.d_1), slice(self.d, self.d_s)

    def _display(self):
        # Display %
        pct = (self.t - self.n - self.s) / (self.T - self.n - self.T % self.s)

        print('{:.2%}'.format(pct), end='\r')
