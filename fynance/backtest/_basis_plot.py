#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-10-24 09:03:49
# @Last modified by: ArthurBernard
# @Last modified time: 2020-10-24 11:08:47

""" Description. """

# Built-in packages

# Third party packages
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# Local packages

__all__ = []


class _BasisFigure(object):
    fig = None
    _n_axes = 0
    _n_cols = 1
    _n_rows = 0

    def __new__(cls, *args, **kwargs):
        if cls.fig is None:
            cls.fig = plt.figure()

        # Set optimal number of axes on figure plot
        cls._n_axes += 1
        if cls._n_cols * cls._n_rows < cls._n_axes:
            if cls._n_rows > cls._n_cols and cls._n_rows >= cls._n_axes ** 0.5:
                cls._n_cols += 1

            else:
                cls._n_rows += 1

        instance = super(_BasisFigure, cls).__new__(cls, *args, **kwargs)

        return instance

    def __del__(self):
        # Set optimal number of axes on figure plot
        _BasisFigure._n_axes -= 1
        n_cols = _BasisFigure._n_cols
        n_rows = _BasisFigure._n_rows
        if (n_cols - 1) * n_rows >= _BasisFigure._n_axis:
            _BasisFigure._n_cols -= 1

        elif (n_rows - 1) * n_cols >= _BasisFigure._n_axis:
            _BasisFigure._n_rows -= 1

        if _BasisFigure._n_axes < 1:
            _BasisFigure.fig = None


class _BasisAxes(_BasisFigure):
    def __init__(self):
        self._num_axes = self._n_axes

    def _set_axes(self):
        self.fig.add_subplots(self._n_rows, self._n_cols, self._num_axes)


if __name__ == "__main__":
    pass
