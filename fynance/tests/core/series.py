#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2021-04-05 20:35:58
# @Last modified by: ArthurBernard
# @Last modified time: 2021-04-05 21:05:28

""" Test of series objects. """

# Built-in packages
from unittest.mock import MagicMock

# Third party packages
import numpy as np
import pytest

# Local packages
from fynance.core.series import Series


__all__ = []


@pytest.fixture
def series():
    return Series([1, 2, 3, 4])


@pytest.fixture
def series_ploted(series):
    mock_ax = MagicMock()
    series.plot(ax=mock_ax)

    return series


class TestSeries:

    def test_constructor(self, series):
        assert series.plot_series is None
        assert str(series) == "[1 2 3 4]"
        assert repr(series) == "Series([1, 2, 3, 4])"
        assert series[0] == 1
        assert series[-1] == 4
        assert len(series) == 4
        assert series.shape == (4,)

    def test_view_casting(self):
        s = np.arange(1, 5).view(Series)
        assert isinstance(s, Series)
        assert s.plot_series is None
        assert repr(s) == "Series([1, 2, 3, 4])"

    def test_new_from_template(self, series_ploted):
        s = series_ploted[1:]
        assert isinstance(s, Series)
        assert s.plot_series is not None
        assert repr(s) == "Series([2, 3, 4])"

    def test_append(self):
        pass

    def test_plot(self):
        pass

    def test_update_plot(self):
        pass

if __name__ == "__main__":
    pass
