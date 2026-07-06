#!/usr/bin/env python3
# coding: utf-8

""" Tests for trade-level analytics (round-trip extraction, trade_summary). """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.metrics.trades import TRADE_DTYPE, extract_trades, trade_summary

# --------------------------------------------------------------------------- #
#   extract_trades -- deterministic paths                                     #
# --------------------------------------------------------------------------- #


def test_extract_trades_flip_and_open_trade():
    # Long run [0, 1], direct flip to short [2, 3] (no flat gap -- two
    # adjacent trades), a flat bar, then an open long trade at the end.
    positions = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 1.0])
    returns = np.array([0.5, 0.5, 0.5, -0.5, 0.0, 0.25])
    out = extract_trades(positions, returns)

    assert out.dtype == TRADE_DTYPE
    assert out.shape[0] == 3

    assert list(out['asset']) == [0, 0, 0]
    assert list(out['t_in']) == [0, 2, 5]
    assert list(out['t_out']) == [1, 3, 5]
    assert list(out['side']) == [1, -1, 1]
    assert list(out['bars']) == [2, 2, 1]
    assert out['ret'][0] == pytest.approx(1.5 * 1.5 - 1.0)
    assert out['ret'][1] == pytest.approx(0.5 * 1.5 - 1.0)
    assert out['ret'][2] == pytest.approx(0.25)


def test_extract_trades_flat_gap_is_not_a_trade():
    positions = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    returns = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    out = extract_trades(positions, returns)
    assert out.shape[0] == 1
    assert out['t_in'][0] == 2
    assert out['t_out'][0] == 2
    assert out['bars'][0] == 1
    assert out['ret'][0] == pytest.approx(0.2)


def test_extract_trades_sized_position_is_one_trade():
    # Constant sign but varying magnitude within the run stays one trade;
    # ret compounds using the actual (sized) position at each bar.
    positions = np.array([0.5, 1.0, 0.25])
    returns = np.array([0.1, 0.1, 0.1])
    out = extract_trades(positions, returns)
    assert out.shape[0] == 1
    assert out['t_in'][0] == 0
    assert out['t_out'][0] == 2
    expected = (1 + 0.5 * 0.1) * (1 + 1.0 * 0.1) * (1 + 0.25 * 0.1) - 1.0
    assert out['ret'][0] == pytest.approx(expected)


def test_extract_trades_all_zero_positions_yields_no_trades():
    positions = np.zeros(10)
    returns = np.zeros(10)
    out = extract_trades(positions, returns)
    assert out.shape[0] == 0
    assert out.dtype == TRADE_DTYPE


def test_extract_trades_2d_book_per_asset():
    # asset 0: long [0,1], flat, long [3]. asset 1: short over the whole book.
    positions = np.array([
        [1.0, -1.0],
        [1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
    ])
    returns = np.array([
        [0.1, 0.1],
        [0.1, -0.1],
        [0.0, 0.2],
        [0.1, 0.1],
    ])
    out = extract_trades(positions, returns)
    assert out.shape[0] == 3

    asset0 = out[out['asset'] == 0]
    asset1 = out[out['asset'] == 1]
    assert list(asset0['t_in']) == [0, 3]
    assert list(asset0['t_out']) == [1, 3]
    assert asset1.shape[0] == 1
    assert asset1['t_in'][0] == 0
    assert asset1['t_out'][0] == 3
    assert asset1['side'][0] == -1


def test_extract_trades_returns_broadcast_1d_over_2d_positions():
    # A single 1-D return path shared by every asset column (the
    # BacktestResult convention).
    positions = np.array([[1.0, -1.0], [1.0, -1.0]])
    returns = np.array([0.1, 0.2])
    out = extract_trades(positions, returns)
    assert out.shape[0] == 2
    asset0 = out[out['asset'] == 0]
    asset1 = out[out['asset'] == 1]
    assert asset0['ret'][0] == pytest.approx(1.1 * 1.2 - 1.0)
    assert asset1['ret'][0] == pytest.approx(0.9 * 0.8 - 1.0)


def test_extract_trades_invalid_shapes_raise():
    with pytest.raises(ValueError):
        extract_trades(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        extract_trades(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError):
        extract_trades(np.zeros((4, 2)), np.zeros((4, 3)))


# --------------------------------------------------------------------------- #
#   extract_trades -- numba vs. slow-python parity                            #
# --------------------------------------------------------------------------- #


def _slow_extract_trades(positions, returns):
    """ Pure-python reference (no numba), mirroring the kernel's own scan. """
    pos = np.asarray(positions, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)
    if ret.ndim == 1:
        ret = np.broadcast_to(ret.reshape(-1, 1), pos.shape)

    T, N = pos.shape
    rows = []
    for j in range(N):
        in_trade = False
        cur_sign = 0
        cur_t_in = 0
        compounded = 1.0
        for t in range(T):
            p = float(pos[t, j])
            if p > 0:
                s = 1
            elif p < 0:
                s = -1
            else:
                s = 0

            if in_trade and s == cur_sign:
                compounded *= (1.0 + p * float(ret[t, j]))
            else:
                if in_trade:
                    rows.append(
                        (j, cur_t_in, t - 1, cur_sign, compounded - 1.0,
                         (t - 1) - cur_t_in + 1)
                    )
                    in_trade = False
                if s != 0:
                    in_trade = True
                    cur_sign = s
                    cur_t_in = t
                    compounded = 1.0 + p * float(ret[t, j])
        if in_trade:
            rows.append(
                (j, cur_t_in, T - 1, cur_sign, compounded - 1.0,
                 (T - 1) - cur_t_in + 1)
            )
    return rows


def test_extract_trades_numba_matches_slow_python_reference():
    rng = np.random.default_rng(42)
    T, N = 500, 3
    # Ternary signal (long/flat/short) so runs, flips and flat gaps all occur.
    positions = rng.choice([-1.0, 0.0, 1.0], size=(T, N), p=[0.3, 0.2, 0.5])
    returns = rng.normal(0.0, 0.01, size=(T, N))

    out = extract_trades(positions, returns)
    ref = _slow_extract_trades(positions, returns)

    assert out.shape[0] == len(ref)
    for row, (asset, t_in, t_out, side, ret, bars) in zip(out, ref):
        assert int(row['asset']) == asset
        assert int(row['t_in']) == t_in
        assert int(row['t_out']) == t_out
        assert int(row['side']) == side
        assert int(row['bars']) == bars
        assert row['ret'] == ret


# --------------------------------------------------------------------------- #
#   trade_summary                                                             #
# --------------------------------------------------------------------------- #


def test_trade_summary_hand_values():
    positions = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 1.0])
    returns = np.array([0.5, 0.5, 0.5, -0.5, 0.0, 0.25])
    trades = extract_trades(positions, returns)
    # rets: [1.25, -0.25, 0.25] -> 2 wins, 1 loss
    s = trade_summary(trades)

    assert s['n_trades'] == 3.0
    assert s['win_rate'] == pytest.approx(2.0 / 3.0)
    assert s['profit_factor'] == pytest.approx((1.25 + 0.25) / 0.25)
    assert s['avg_win'] == pytest.approx((1.25 + 0.25) / 2.0)
    assert s['avg_loss'] == pytest.approx(-0.25)
    assert s['payoff_ratio'] == pytest.approx(s['avg_win'] / abs(s['avg_loss']))
    assert s['expectancy'] == pytest.approx((1.25 - 0.25 + 0.25) / 3.0)
    assert s['mean_bars'] == pytest.approx((2 + 2 + 1) / 3.0)
    assert s['median_bars'] == pytest.approx(2.0)


def test_trade_summary_streaks():
    # ret sequence: win, win, loss, win, loss, loss, loss -> max_win=2, max_loss=3
    trades = np.zeros(7, dtype=TRADE_DTYPE)
    trades['ret'] = [0.1, 0.2, -0.1, 0.05, -0.2, -0.1, -0.3]
    trades['bars'] = 1
    s = trade_summary(trades)
    assert s['max_win_streak'] == 2.0
    assert s['max_loss_streak'] == 3.0


def test_trade_summary_no_losses_is_inf_profit_factor():
    trades = np.zeros(2, dtype=TRADE_DTYPE)
    trades['ret'] = [0.1, 0.2]
    trades['bars'] = 1
    s = trade_summary(trades)
    assert s['profit_factor'] == float('inf')
    assert np.isnan(s['payoff_ratio'])  # no losses to compare against
    assert np.isnan(s['avg_loss'])


def test_trade_summary_empty_is_zeros_and_nan():
    empty = np.empty(0, dtype=TRADE_DTYPE)
    s = trade_summary(empty)

    assert s['n_trades'] == 0.0
    assert s['win_rate'] == 0.0
    assert s['max_win_streak'] == 0.0
    assert s['max_loss_streak'] == 0.0
    for key in ('profit_factor', 'avg_win', 'avg_loss', 'payoff_ratio',
                'expectancy', 'mean_bars', 'median_bars'):
        assert np.isnan(s[key])
