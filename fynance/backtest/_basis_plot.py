#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-10-24 09:03:49
# @Last modified by: ArthurBernard
# @Last modified time: 2020-11-20 09:09:02

""" Description. """

# Built-in packages
from abc import ABCMeta, abstractmethod

# Third party packages
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# Local packages

__all__ = []


class _BasisAxes(metaclass=ABCMeta):
    def __call__(self, fig, n_rows, n_cols, n_axes):
        self.fig = fig
        self.n_axes = n_axes
        self.ax = self.fig.add_subplot(n_rows, n_cols, n_axes)

    # @abstractmethod
    def plot(self, y, x=None):
        if x is None:
            self.ax.plot(y)

        else:
            self.ax.plot(x, y)


class _BasisPlot:
    def __init__(self, **kwargs):
        self.fig = plt.figure(**kwargs)
        self._n_axes = 0
        self._n_cols = 1
        self._n_rows = 0
        self.axes = {}
        self.keys = []

    def __setitem__(self, key, value: _BasisAxes):
        # Set optimal number of axes on figure plot
        self._n_axes += 1
        sqrt_n_axes = self._n_axes ** 0.5
        if self._n_cols * self._n_rows < self._n_axes:
            if self._n_rows > self._n_cols and self._n_rows >= sqrt_n_axes:
                self._n_cols += 1

            else:
                self._n_rows += 1

        print(self._n_rows, self._n_cols)
        self.axes[key] = value
        self.keys.append(key)

    def __delitem__(self, key):
        # Set optimal number of axes on figure plot
        self._n_axes -= 1
        n_cols = self._n_cols
        n_rows = self._n_rows
        if (n_cols - 1) * n_rows >= self._n_axes:
            self._n_cols -= 1

        elif (n_rows - 1) * n_cols >= self._n_axes:
            self._n_rows -= 1

        if self._n_axes < 1:
            self.fig = None
            print("destruct fig")

        del self.axes[key]
        self.keys.remove(key)

    def __getitem__(self, key):
        return self.axes[key]

    def set_axes(self):
        for i, key in enumerate(self.keys, 1):
            self.axes[key](self.fig, self._n_rows, self._n_cols, i)

        return self


if __name__ == "__main__":
    pass
