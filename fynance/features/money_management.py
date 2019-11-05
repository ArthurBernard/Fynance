#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-03-15 12:23:04
# @Last modified by: ArthurBernard
# @Last modified time: 2019-11-05 15:56:01

""" Module with function to compute some money management coefficients. """

# Built-in packages

# External packages
import numpy as np

# Internal packages
from fynance.features.momentums import ema

__all__ = ['iso_vol']

# =========================================================================== #
#                         Money-management Tools                              #
# =========================================================================== #


def iso_vol(series, target_vol=0.20, leverage=1., period=252, half_life=11):
    """ Make an iso-vol vector to apply to signal vector.

    Parameters
    ----------
    series : np.ndarray[ndim=1, dtype=np.float64]
        Series of price of underlying.
    target_vol : float (default 20 %)
        Volatility to target.
    leverage : float (default 1)
        Max leverage to use.
    period : int (default 250)
        Number of period per year.
    half_life : int (default 11)
        Half-life of exponential moving average used to compute volatility.

    Returns
    -------
    iv : np.ndarray[ndim=1, dtype=np.float64]
        Series of iso-vol coefficient.

    Examples
    --------
    >>> series = np.array([95, 100, 85, 105, 110, 90]).astype(np.float64)
    >>> iso_vol(series, target_vol=0.5, leverage=2, period=12, half_life=3)
    array([1.        , 1.        , 2.        , 1.11289534, 0.88580571,
           1.20664917])

    """
    # Set iso-vol vector
    iv = np.ones([series.size])
    # Compute squared daily return vector
    ret2 = np.square(series[:-1] / series[1:] - 1)
    # Compute volatility vector
    vol = np.sqrt(period * ema(ret2, w=half_life))
    vol[vol <= 0.] = 1e-8
    # Compute iso-vol coefficient
    iv[2:] = target_vol / vol[:-1]
    # Cap with the max leverage available
    iv[iv > leverage] = leverage
    return iv
