#!/usr/bin/env python3
# coding: utf-8

# Built-in packages

# External packages
import numpy as np

# Internal packages
from fynance.tools.metrics import accuracy, sharpe, calmar

__all__ = ['set_text_stats']

# =========================================================================== #
#                              Printer Tools                                  #
# =========================================================================== #


def set_text_stats(underly, period=252, accur=True, perf=True, vol=True,
                   sharp=True, calma=True, underlying='Underlying', fees=0,
                   **kwpred):
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
        perf = np.exp(np.cumsum(underly))
        perf_targ = np.sign(perf[-1] / perf[0]) * np.float_power(
            np.abs(perf[-1] / perf[0]), period / perf.size) - 1.
        txt += '| {:14} | {:10.2%} |\n'.format(underlying, perf_targ)
        for key, pred in kwpred.items():
            vect_fee = np.zeros(pred.shape)
            vect_fee[1:] += np.abs(pred[1:] - pred[:-1]) * fees
            perf = np.exp(np.cumsum(underly * pred - vect_fee))
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
            vect_fee = np.zeros(pred.shape)
            vect_fee[1:] += np.abs(pred[1:] - pred[:-1]) * fees
            perf = np.exp(np.cumsum(underly * pred - vect_fee))
            vol_pred = np.sqrt(period) * np.std(perf[1:] / perf[:-1] - 1)
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
            vect_fee[1:] += np.abs(pred[1:] - pred[:-1]) * fees
            perf = np.exp(np.cumsum(underly * pred - vect_fee))
            sharpe_pred = sharpe(perf, period=period)
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
            vect_fee[1:] += np.abs(pred[1:] - pred[:-1]) * fees
            perf = np.exp(np.cumsum(underly * pred - vect_fee))
            calmar_pred = calmar(perf, period=period)
            txt += '| {:14} | {:10.2f} |\n'.format(key, calmar_pred)
    txt += '+=============================+\n'
    return txt
