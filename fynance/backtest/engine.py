#!/usr/bin/env python3
# coding: utf-8

""" Vectorized backtest engine.

The core artery of the library: turn a position book + asset returns (or
prices) + a cost model into a :class:`~fynance.backtest.result.BacktestResult`.
Pure numpy, strictly causal.

"""

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import numpy as np

# Local packages
from fynance.backtest.result import BacktestResult

__all__ = ['backtest']


def backtest(
    data: Any,
    positions: Any,
    cost: Any = None,
    capital: float = 1.0,
    returns_input: bool = True,
    shift: bool = True,
) -> BacktestResult:
    """ Run a vectorized backtest.

    Parameters
    ----------
    data : array-like
        Asset returns (default) or prices (``returns_input=False``). Shape
        ``(T,)`` for a single asset or ``(T, n_assets)`` for a book.
    positions : array-like
        Position/weight series aligned with ``data``.
    cost : CostModel, optional
        Per-step transaction cost model (e.g.
        :class:`~fynance.backtest.cost.ProportionalCost`).
    capital : float
        Initial capital.
    returns_input : bool
        If ``True`` (default) ``data`` are returns; otherwise prices (converted
        to pct returns; the first observation is dropped).
    shift : bool
        If ``True`` (default) positions are shifted one step (the position
        decided at ``t`` earns the return at ``t+1``), guaranteeing causality.

    Returns
    -------
    BacktestResult
        Equity, net/gross returns, positions and per-step costs.

    """
    arr = np.asarray(data, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)

    # ``cost_book`` is the full position book the cost model sees: it must
    # retain the initial trade so turnover is charged from the first entry. On
    # a prices input the trailing position earns no return and is dropped, but
    # its turnover is never incurred either, so we drop it from the cost book
    # too (charging only trades into positions that actually earn a return).
    cost_book = pos

    if returns_input:
        returns = arr

    else:
        returns = arr[1:] / arr[:-1] - 1.0

        if pos.shape[0] == arr.shape[0]:
            # Drop the first position (it predates the first return): the
            # decision at price index k earns the return over [k, k+1].
            pos = pos[1:]
            # The last decision earns no return; charge turnover up to it only.
            cost_book = cost_book[:-1]

    if pos.shape[0] != returns.shape[0]:

        raise ValueError(
            f"positions length {pos.shape[0]} != returns length "
            f"{returns.shape[0]}"
        )

    # Causal shift: position decided at t earns the return at t+1.
    if shift:
        pos_eff = np.zeros_like(pos)
        pos_eff[1:] = pos[:-1]

    else:
        pos_eff = pos

    gross = pos_eff * returns

    if gross.ndim == 2:
        gross = gross.sum(axis=1)

    costs = (
        np.asarray(cost(cost_book), dtype=np.float64)
        if cost is not None
        else np.zeros(returns.shape[0], dtype=np.float64)
    )
    net = gross - costs
    equity = capital * np.cumprod(1.0 + net)

    return BacktestResult(
        equity=equity,
        returns=net,
        gross_returns=gross,
        positions=pos,
        costs=costs,
    )
