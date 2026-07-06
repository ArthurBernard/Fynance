#!/usr/bin/env python3
# coding: utf-8

""" Trade-level analytics — round-trip extraction and summary statistics.

Where :mod:`fynance.metrics.trading` describes *churn* (how often the position
flips), this module describes individual **round-trip trades**: it segments a
position series into maximal runs of constant nonzero sign and reports, per
run, the entry/exit bars, side and compounded net return. :func:`trade_summary`
then aggregates a set of such trades into the usual trading-desk statistics
(win rate, profit factor, streaks, ...).

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ['extract_trades', 'trade_summary']

#: Structured dtype of the array returned by :func:`extract_trades`.
TRADE_DTYPE = np.dtype([
    ('asset', np.int16), ('t_in', np.int64), ('t_out', np.int64),
    ('side', np.int8), ('ret', np.float64), ('bars', np.int64),
])


# --------------------------------------------------------------------------- #
#   numba kernel                                                              #
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _extract_trades_kernel(positions, returns):
    """ Scan one column and extract its maximal constant-sign runs.

    A run breaks whenever ``sign(positions[t])`` changes -- including a
    transition through/from ``0`` (flat) -- so a direct long -> short flip
    with no flat bar in between yields two immediately adjacent trades (the
    first's ``t_out`` is the bar right before the second's ``t_in``), rather
    than being merged or skipped. ``NaN`` compares false to both ``> 0`` and
    ``< 0``, so a ``NaN`` position is treated as flat (breaks any open run).
    A run still open at the last bar is closed there and returned as-is
    (see the ``extract_trades`` docstring).
    """
    T = positions.shape[0]
    t_in_out = np.empty(T, dtype=np.int64)
    t_out_out = np.empty(T, dtype=np.int64)
    side_out = np.empty(T, dtype=np.int8)
    ret_out = np.empty(T, dtype=np.float64)
    bars_out = np.empty(T, dtype=np.int64)

    n = 0
    in_trade = False
    cur_sign = np.int8(0)
    cur_t_in = 0
    compounded = 1.0

    for t in range(T):
        s = np.int8(0)
        if positions[t] > 0:
            s = np.int8(1)
        elif positions[t] < 0:
            s = np.int8(-1)

        if in_trade and s == cur_sign:
            compounded *= (1.0 + positions[t] * returns[t])
        else:
            if in_trade:
                t_in_out[n] = cur_t_in
                t_out_out[n] = t - 1
                side_out[n] = cur_sign
                ret_out[n] = compounded - 1.0
                bars_out[n] = (t - 1) - cur_t_in + 1
                n += 1
                in_trade = False

            if s != 0:
                in_trade = True
                cur_sign = s
                cur_t_in = t
                compounded = 1.0 + positions[t] * returns[t]

    if in_trade:
        t_in_out[n] = cur_t_in
        t_out_out[n] = T - 1
        side_out[n] = cur_sign
        ret_out[n] = compounded - 1.0
        bars_out[n] = (T - 1) - cur_t_in + 1
        n += 1

    return t_in_out[:n], t_out_out[:n], side_out[:n], ret_out[:n], bars_out[:n]


# --------------------------------------------------------------------------- #
#   public API                                                                 #
# --------------------------------------------------------------------------- #


def extract_trades(positions: NDArray, returns: NDArray) -> NDArray:
    r""" Extract round-trip trades from a position series.

    A trade is a maximal run of bars over which ``sign(positions)`` is
    constant and nonzero -- flat (``0``) bars are never part of a trade and
    always close any open run. Its realized return is the compounded net
    return of holding the (possibly sized, not just signed) position over the
    run:

    .. math::

        ret = \prod_{t=t_{in}}^{t_{out}} \left(1 + positions_t \cdot
              returns_t\right) - 1

    ``positions`` and ``returns`` are assumed already aligned index-for-index
    (``positions[t]`` earns ``returns[t]``) -- the convention
    :class:`~fynance.backtest.result.BacktestResult` stores them in, so
    ``extract_trades(result.positions, result.returns)`` and
    :meth:`~fynance.backtest.result.BacktestResult.trades` agree exactly (see
    that class for how the causal shift is folded into ``returns`` upstream,
    in the engine).

    Parameters
    ----------
    positions : np.ndarray[dtype, ndim=1 or 2]
        Position / weight series, shape ``(T,)`` for a single asset or
        ``(T, n_assets)`` for a book (column-wise, one independent scan per
        column).
    returns : np.ndarray[dtype, ndim=1 or 2]
        Realized return series aligned with ``positions``. Either the same
        shape as ``positions``, or, for a 2-D ``positions``, a 1-D array of
        shape ``(T,)`` broadcast to every column (the
        :class:`~fynance.backtest.result.BacktestResult` case: a single net
        return path shared by every asset's own position run).

    Returns
    -------
    np.ndarray[TRADE_DTYPE, ndim=1]
        One row per trade, ordered by asset then by time within each asset,
        with fields:

        - ``asset`` (``int16``) -- column index (``0`` for a 1-D
          ``positions``).
        - ``t_in`` (``int64``) -- first bar of the run.
        - ``t_out`` (``int64``) -- last bar of the run, inclusive. A run
          still open at the last observation is included with its current
          (unrealized) mark rather than dropped.
        - ``side`` (``int8``) -- ``+1`` long, ``-1`` short.
        - ``ret`` (``float64``) -- compounded net return over the run.
        - ``bars`` (``int64``) -- ``t_out - t_in + 1``.

    Raises
    ------
    ValueError
        If ``positions`` or ``returns`` is not 1-D or 2-D, or their shapes
        are incompatible (see above).

    Examples
    --------
    A long run of two bars, a direct flip to a short run of two bars (no flat
    gap -- two adjacent trades, not one), a flat bar, then an open long trade
    at the end:

    >>> import numpy as np
    >>> positions = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 1.0])
    >>> returns = np.array([0.5, 0.5, 0.5, -0.5, 0.0, 0.25])
    >>> trades = extract_trades(positions, returns)
    >>> trades['t_in'], trades['t_out'], trades['side']
    (array([0, 2, 5]), array([1, 3, 5]), array([ 1, -1,  1], dtype=int8))
    >>> trades['ret']
    array([ 1.25, -0.25,  0.25])
    >>> trades['bars']
    array([2, 2, 1])

    A two-asset book: each column is scanned independently and tagged with
    its own ``asset`` index:

    >>> pos2 = np.array([[1.0, -1.0], [1.0, -1.0], [0.0, -1.0]])
    >>> ret2 = np.array([[0.1, 0.1], [0.1, -0.1], [0.0, 0.2]])
    >>> trades2 = extract_trades(pos2, ret2)
    >>> trades2['asset']
    array([0, 1], dtype=int16)
    >>> trades2['t_in'], trades2['t_out']
    (array([0, 0]), array([1, 2]))

    See Also
    --------
    trade_summary
    fynance.metrics.trading.sign_changes, fynance.metrics.trading.trades_per_year
    fynance.backtest.result.BacktestResult.trades

    """
    pos = np.asarray(positions, dtype=np.float64)

    if pos.ndim == 1:
        pos2d = pos.reshape(-1, 1)
    elif pos.ndim == 2:
        pos2d = pos
    else:

        raise ValueError(f"positions must be 1-D or 2-D, got ndim={pos.ndim}")

    T, N = pos2d.shape
    ret = np.asarray(returns, dtype=np.float64)

    if ret.ndim == 1:
        if ret.shape[0] != T:

            raise ValueError(
                f"returns length {ret.shape[0]} != positions length {T}"
            )
        ret2d = np.broadcast_to(ret.reshape(-1, 1), (T, N))
    elif ret.ndim == 2:
        if ret.shape != pos2d.shape:

            raise ValueError(
                f"returns shape {ret.shape} != positions shape {pos2d.shape}"
            )
        ret2d = ret
    else:

        raise ValueError(f"returns must be 1-D or 2-D, got ndim={ret.ndim}")

    chunks = []
    for j in range(N):
        t_in, t_out, side, r, bars = _extract_trades_kernel(
            np.ascontiguousarray(pos2d[:, j]), np.ascontiguousarray(ret2d[:, j])
        )
        chunk = np.empty(t_in.shape[0], dtype=TRADE_DTYPE)
        chunk['asset'] = 0 if pos.ndim == 1 else j
        chunk['t_in'] = t_in
        chunk['t_out'] = t_out
        chunk['side'] = side
        chunk['ret'] = r
        chunk['bars'] = bars
        chunks.append(chunk)

    if chunks:

        return np.concatenate(chunks)

    return np.empty(0, dtype=TRADE_DTYPE)


def _max_streak(mask: NDArray) -> int:
    """ Longest run of consecutive ``True`` values in a 1-D boolean array. """
    best = run = 0
    for v in mask:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0

    return best


def trade_summary(trades: NDArray) -> dict[str, float]:
    r""" Summary statistics of a set of round-trip trades.

    Parameters
    ----------
    trades : np.ndarray[TRADE_DTYPE, ndim=1]
        Output of :func:`extract_trades` (or any array sharing its dtype and
        field semantics). ``max_win_streak`` / ``max_loss_streak`` are
        computed over ``trades`` in the order given, so pass a single
        asset's trades (already time-ordered) for a meaningful streak --
        e.g. ``trades[trades['asset'] == k]``.

    Returns
    -------
    dict of str to float
        - ``n_trades`` -- number of trades.
        - ``win_rate`` -- share of trades with ``ret > 0`` (of all trades,
          not just wins + losses).
        - ``profit_factor`` -- ``sum(wins) / abs(sum(losses))``; ``inf`` if
          there are winning trades and no losing ones; ``NaN`` if there are
          no trades or none with a nonzero return either way.
        - ``avg_win``, ``avg_loss`` -- mean ``ret`` of winning / losing
          trades (``avg_loss`` is negative); ``NaN`` if that side is empty.
        - ``payoff_ratio`` -- ``avg_win / abs(avg_loss)``; ``NaN`` unless
          both sides are non-empty.
        - ``expectancy`` -- mean ``ret`` over *all* trades.
        - ``max_win_streak``, ``max_loss_streak`` -- longest run of
          consecutive winning / losing trades (a breakeven ``ret == 0``
          trade breaks both streaks).
        - ``mean_bars``, ``median_bars`` -- mean / median holding period.

        ``trades`` empty: ``n_trades``, ``win_rate``, ``max_win_streak`` and
        ``max_loss_streak`` are ``0.0``; every other field is ``NaN`` (there
        is nothing to average).

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 1.0])
    >>> returns = np.array([0.5, 0.5, 0.5, -0.5, 0.0, 0.25])
    >>> trades = extract_trades(positions, returns)
    >>> s = trade_summary(trades)
    >>> s['n_trades'], s['win_rate']
    (3.0, 0.6666666666666666)
    >>> round(s['profit_factor'], 4)
    6.0
    >>> s['max_win_streak'], s['max_loss_streak']
    (1.0, 1.0)

    An empty set of trades reports zero counts and ``NaN`` averages:

    >>> empty = extract_trades(np.zeros(4), np.zeros(4))
    >>> trade_summary(empty)['n_trades'], trade_summary(empty)['profit_factor']
    (0.0, nan)

    See Also
    --------
    extract_trades

    """
    n = trades.shape[0]

    if n == 0:

        return {
            'n_trades': 0.0,
            'win_rate': 0.0,
            'profit_factor': float('nan'),
            'avg_win': float('nan'),
            'avg_loss': float('nan'),
            'payoff_ratio': float('nan'),
            'expectancy': float('nan'),
            'max_win_streak': 0.0,
            'max_loss_streak': 0.0,
            'mean_bars': float('nan'),
            'median_bars': float('nan'),
        }

    ret = trades['ret'].astype(np.float64)
    bars = trades['bars'].astype(np.float64)
    win_mask = ret > 0.0
    loss_mask = ret < 0.0
    wins = ret[win_mask]
    losses = ret[loss_mask]

    sum_wins = float(wins.sum()) if wins.size else 0.0
    sum_losses = float(losses.sum()) if losses.size else 0.0
    abs_losses = abs(sum_losses)

    if abs_losses > 0.0:
        profit_factor = sum_wins / abs_losses
    elif sum_wins > 0.0:
        profit_factor = float('inf')
    else:
        profit_factor = float('nan')

    avg_win = float(wins.mean()) if wins.size else float('nan')
    avg_loss = float(losses.mean()) if losses.size else float('nan')
    payoff_ratio = (
        avg_win / abs(avg_loss) if wins.size and losses.size else float('nan')
    )

    return {
        'n_trades': float(n),
        'win_rate': float(win_mask.sum()) / n,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'payoff_ratio': payoff_ratio,
        'expectancy': float(ret.mean()),
        'max_win_streak': float(_max_streak(win_mask)),
        'max_loss_streak': float(_max_streak(loss_mask)),
        'mean_bars': float(bars.mean()),
        'median_bars': float(np.median(bars)),
    }


if __name__ == '__main__':

    import doctest

    doctest.testmod()
