#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-11-05 15:50:33
# @Last modified by: ArthurBernard
# @Last modified time: 2020-05-08 09:22:17

from __future__ import annotations

# Built-in packages
# Third party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.features.metrics import accuracy, calmar, sharpe

__all__ = ['set_text_stats']

# =========================================================================== #
#                              Printer Tools                                  #
# =========================================================================== #


def set_text_stats(
    underly: NDArray,
    period: int = 252,
    accur: bool = True,
    perf: bool = True,
    vol: bool = True,
    sharp: bool = True,
    calma: bool = True,
    underlying: str = 'Underlying',
    fees: float = 0,
    **kwpred,
) -> str:
    """ Set a table as string with different indicators (accuracy, perf, vol
    and sharpe) for underlying and several strategies.

    Parameters
    ----------
    underly : np.ndarray[ndim=1, dtype=np.float64]
        Series of underlying prices.
    period : int, optional
        Number of period per day, default is 252.
    accur : bool, optional
        If true compute accuracy else not, default is True.
    perf : bool, optional
        If true compute performance else not, default is True.
    vol : bool, optional
        If true compute volatility else not, default is True.
    sharp : bool, optional
        If true compute sharpe ratio else not, default is True.
    calma : bool, optional
        If true compute calmar ratio else not, default is True.
    underlying : str, optional
        Name of the underlying, default is 'Underlying'.
    kwpred : dict of np.ndarray
        Any strategies or predictions that you want to compare.
    fees : float, optional
        Fees to apply at the strategy performance.

    Returns
    -------
    txt : str
        Table of results.

    See Also
    --------
    PlotBackTest, display_perf

    """
    txt = ''
    # Compute Accuracy
    if accur:
        txt += '+=============================+\n'
        txt += '|          Accuracy           |\n'
        txt += '+----------------+------------+\n'
        for key, pred in kwpred.items():
            accu_pred = accuracy(underly, pred)
            txt += '| {:14} | {:10.2%} |\n'.format(key, accu_pred)
    # Compute performance
    if perf:
        txt += '+=============================+\n'
        txt += '|         Performance         |\n'
        txt += '+----------------+------------+\n'
        cum_perf = np.exp(np.cumsum(underly))
        perf_targ = np.sign(cum_perf[-1] / cum_perf[0]) * np.float_power(
            np.abs(cum_perf[-1] / cum_perf[0]), period / cum_perf.size) - 1.
        txt += '| {:14} | {:10.2%} |\n'.format(underlying, perf_targ)
        for key, pred in kwpred.items():
            vect_fee = np.zeros(pred.shape)
            vect_fee[1:] += (pred[1:] - pred[:-1]) * fees
            cum_perf = np.cumprod((np.exp(underly) - 1 - vect_fee) * pred + 1)
            perf_pred = np.sign(cum_perf[-1] / cum_perf[0]) * np.float_power(
                np.abs(cum_perf[-1] / cum_perf[0]), period / cum_perf.size) - 1.
            txt += '| {:14} | {:10.2%} |\n'.format(key, perf_pred)
    # Compute volatility
    if vol:
        txt += '+=============================+\n'
        txt += '|          Volatility         |\n'
        txt += '+----------------+------------+\n'
        cum_perf = np.exp(np.cumsum(underly))
        vol_targ = np.sqrt(period) * np.std(cum_perf[1:] / cum_perf[:-1] - 1)
        txt += '| {:14} | {:10.2%} |\n'.format(underlying, vol_targ)
        for key, pred in kwpred.items():
            vect_fee = np.zeros(pred.shape)
            vect_fee[1:] += (pred[1:] - pred[:-1]) * fees
            cum_perf = np.cumprod((np.exp(underly) - 1 - vect_fee) * pred + 1)
            vol_pred = np.sqrt(period) * np.std(cum_perf[1:] / cum_perf[:-1] - 1)
            txt += '| {:14} | {:10.2%} |\n'.format(key, vol_pred)
    # Compute sharpe Ratio
    if sharp:
        txt += '+=============================+\n'
        txt += '|         Sharpe Ratio        |\n'
        txt += '+----------------+------------+\n'
        sharpe_targ = sharpe(np.exp(np.cumsum(underly)), period=period)
        txt += '| {:14} | {:10.2f} |\n'.format(underlying, sharpe_targ)
        for key, pred in kwpred.items():
            vect_fee = np.zeros(pred.shape)
            vect_fee[1:] += (pred[1:] - pred[:-1]) * fees
            cum_perf = np.cumprod((np.exp(underly) - 1 - vect_fee) * pred + 1)
            sharpe_pred = sharpe(cum_perf, period=period)
            txt += '| {:14} | {:10.2f} |\n'.format(key, sharpe_pred)
    # Compute calmar
    if calma:
        txt += '+=============================+\n'
        txt += '|         Calmar Ratio        |\n'
        txt += '+----------------+------------+\n'
        calmar_targ = calmar(np.exp(np.cumsum(underly)), period=period)
        txt += '| {:14} | {:10.2f} |\n'.format(underlying, calmar_targ)
        for key, pred in kwpred.items():
            vect_fee = np.zeros(pred.shape)
            vect_fee[1:] += (pred[1:] - pred[:-1]) * fees
            cum_perf = np.cumprod((np.exp(underly) - 1 - vect_fee) * pred + 1)
            calmar_pred = calmar(cum_perf, period=period)
            txt += '| {:14} | {:10.2f} |\n'.format(key, calmar_pred)
    txt += '+=============================+\n'
    return txt
