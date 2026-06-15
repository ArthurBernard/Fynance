#!/usr/bin/env python3
# coding: utf-8

""" Pure (Streamlit-free) helpers for the playground.

Separated from the UI so the signal-running logic is unit-testable without a
Streamlit runtime.

"""

from __future__ import annotations

# Built-in packages
from typing import Any, Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.backtest.result import BacktestResult
from fynance.strategy import Strategy

__all__ = ['run_signal', 'compile_signal', 'TEMPLATE']

TEMPLATE = '''\
import numpy as np

def signal(prices):
    """Return a position series (in [-1, 1]) aligned with `prices`.

    Example: long when price is above its 20-period moving average.
    """
    ma = np.convolve(prices, np.ones(20) / 20, mode="same")
    return np.where(prices > ma, 1.0, -1.0)
'''


def compile_signal(code: str) -> Callable[[NDArray], NDArray]:
    """ Compile user code defining a ``signal(prices)`` function.

    Executes ``code`` (local-only tool; no sandbox) and returns its ``signal``
    callable.
    """
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102 (local-only playground)

    if "signal" not in namespace or not callable(namespace["signal"]):

        raise ValueError("the code must define a callable `signal(prices)`")

    return namespace["signal"]


def run_signal(prices: NDArray, signal: Callable[[NDArray], NDArray],
               fee: float = 0.0) -> BacktestResult:
    """ Backtest a ``signal(prices) -> positions`` function on a price series. """
    from fynance.backtest.cost import ProportionalCost

    cost = ProportionalCost(fee=fee) if fee else None
    strat = Strategy(features=lambda p: np.asarray(signal(p), dtype=np.float64),
                     signal=lambda x: x, cost=cost)

    return strat.run(np.asarray(prices, dtype=np.float64))
