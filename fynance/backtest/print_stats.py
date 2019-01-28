#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np

# Internal packages
from fynance.tools.metrics import accuracy, sharpe, calmar

__all__ = ['set_text_stats']

#=============================================================================#
#                              Printer Tools                                  #
#=============================================================================#


def set_text_stats(
        underly, period=252, accur=True, perf=True, vol=True, sharpe=True, 
        calmar=True, underlying='Underlying', **kwpred
    ):
    """ 
    Set a table as string with different indicators (accuracy, perf, vol and 
    sharpe) for underlying and several strategies. 
    
    Parameters
    ----------
    :underly: np.ndarray[ndim=1, dtype=np.float64]
        Series of underlying prices.
    :period: int (default 252)
        Number of period per day.
    :accur: bool (default is True)
        If true compute accuracy else not.
    :perf: bool (default is True)
        If true compute performance else not.
    :vol: bool (default is True)
        If true compute volatility else not.
    :sharpe: bool (default is True)
        If true compute sharpe ratio else not.
    :calmar: bool (default is True)
        If true compute calmar ratio else not.
    :underlying: str (default is 'Underlying')
        Name of the underlying.
    :kwpred: Any strategies or predictions that you want to compare.

    Return
    ------
    :txt: str
        Table of results.
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
        perf = np.exp(np.cumsum(underly))
        perf_targ = np.sign(perf[-1] / perf[0]) * np.float_power(
            np.abs(perf[-1] / perf[0]), period / perf.size) - 1.
        txt += '| {:14} | {:10.2%} |\n'.format(underlying, perf_targ)
        for key, pred in kwpred.items():
            perf = np.exp(np.cumsum(underly * pred))
            perf_pred = np.sign(perf[-1] / perf[0]) * np.float_power(
                np.abs(perf[-1] / perf[0]), period / perf.size) - 1.
            txt += '| {:14} | {:10.2%} |\n'.format(key, perf_pred)
    # Compute volatility
    if vol:
        txt += '+=============================+\n'
        txt += '|          Volatility         |\n'
        txt += '+----------------+------------+\n'
        perf = np.exp(np.cumsum(underly))
        vol_targ = np.sqrt(period) * np.std(perf[1:] / perf[:-1] - 1)
        txt += '| {:14} | {:10.2%} |\n'.format(underlying, vol_targ)
        for key, pred in kwpred.items():
            perf = np.exp(np.cumsum(underly * pred))
            vol_pred = np.sqrt(period) * np.std(perf[1:] / perf[:-1] - 1)
            txt += '| {:14} | {:10.2%} |\n'.format(key, vol_pred)
    # Compute sharpe Ratio
    if sharpe:
        txt += '+=============================+\n'
        txt += '|         Sharpe Ratio        |\n'
        txt += '+----------------+------------+\n'
        sharpe_targ = sharpe(np.exp(np.cumsum(underly)), period=period)
        txt += '| {:14} | {:10.2f} |\n'.format(underlying, sharpe_targ)
        for key, pred in kwpred.items():
            sharpe_pred = sharpe(
                np.exp(np.cumsum(underly * pred)), period=period
            )
            txt += '| {:14} | {:10.2f} |\n'.format(key, sharpe_pred)
    # Compute calmar
    if calmar:
        txt += '+=============================+\n'
        txt += '|         Calmar Ratio        |\n'
        txt += '+----------------+------------+\n'
        calmar_targ = calmar(np.exp(np.cumsum(underly)), period=period)
        txt += '| {:14} | {:10.2f} |\n'.format(underlying, calmar_targ)
        for key, pred in kwpred.items():
            calmar_pred = calmar(
                np.exp(np.cumsum(underly * pred)), period=period
            )
            txt += '| {:14} | {:10.2f} |\n'.format(key, calmar_pred)
    txt += '+=============================+\n'
    return txt