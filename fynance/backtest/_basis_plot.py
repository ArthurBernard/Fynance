#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-10-24 09:03:49
# @Last modified by: ArthurBernard
# @Last modified time: 2020-11-20 08:55:36

""" Description. """

# Built-in packages
from abc import ABCMeta, abstractmethod

# Third party packages
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# Local packages

__all__ = []


class MetaFigure(type):
    def __new__(mcls, name, bases, attrs):
        attrs['fig'] = None
        attrs['_n_axes'] = 0
        attrs['_n_cols'] = 1
        attrs['_n_rows'] = 0

        return super(MetaFigure, mcls).__new__(mcls, name, bases, attrs)


class _BasisFigure2(metaclass=MetaFigure):
    def __new__(cls, *args, **kwargs):
        if cls.fig is None:
            cls.fig = plt.figure()
            print("create fig")

        # Set optimal number of axes on figure plot
        cls._n_axes += 1
        if cls._n_cols * cls._n_rows < cls._n_axes:
            if cls._n_rows > cls._n_cols and cls._n_rows >= cls._n_axes ** 0.5:
                cls._n_cols += 1

            else:
                cls._n_rows += 1

        instance = super(_BasisFigure2, cls).__new__(cls, *args, **kwargs)
        instance

        instance._num_axes = cls._n_axes
        instance.ax = cls.fig.add_subplot(
            cls._n_rows,
            cls._n_cols,
            instance._num_axes
        )
        print("add {}e axes".format(instance._num_axes))

        return instance

    def __init__(self):
        pass

    def __del__(self):
        # Set optimal number of axes on figure plot
        _BasisFigure2._n_axes -= 1
        n_cols = _BasisFigure2._n_cols
        n_rows = _BasisFigure2._n_rows
        if (n_cols - 1) * n_rows >= _BasisFigure2._n_axes:
            _BasisFigure2._n_cols -= 1

        elif (n_rows - 1) * n_cols >= _BasisFigure2._n_axes:
            _BasisFigure2._n_rows -= 1

        if _BasisFigure2._n_axes < 1:
            _BasisFigure2.fig = None
            print("destruct fig")

    def plot(self, y, x=None):
        if x is None:
            self.ax.plot(y)

        else:
            self.ax.plot(x, y)


class _BasisFigure(metaclass=MetaFigure):
    def __new__(cls, *args, **kwargs):
        if cls.fig is None:
            cls.fig = plt.figure()
            print("create fig")

        instance = super(_BasisFigure, cls).__new__(cls, *args, **kwargs)

        return instance

    def __init__(self):
        # Set optimal number of axes on figure plot
        self._n_axes += 1
        if self._n_cols * self._n_rows < self._n_axes:
            if self._n_rows > self._n_cols and self._n_rows >= self._n_axes ** 0.5:
                self._n_cols += 1

            else:
                self._n_rows += 1

    def __del__(self):
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


class _BasisAxes2(_BasisFigure):
    def __init__(self):
        super(_BasisAxes, self).__init__()
        self._num_axes = self._n_axes
        print("{}e axes".format(self._num_axes))
        self._set_axes()

    def __del__(self):
        super(_BasisAxes, self).__del__()

    def _set_axes(self):
        # Append a new axis on figure
        self.ax = self.fig.add_subplot(
            self._n_rows,
            self._n_cols,
            self._num_axes
        )

    def plot(self, y, x=None):
        if x is None:
            self.ax.plot(y)

        else:
            self.ax.plot(x, y)


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
