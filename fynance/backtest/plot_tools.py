#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Internal packages
from fynance.tools.money_management import iso_vol
from fynance.tools.metrics import drawdown, roll_sharpe

# Set plot style
plt.style.use('seaborn')

__all__ = ['display_perf']

#=============================================================================#
#                              Printer Tools                                  #
#=============================================================================#


def display_perf(
        y_idx, y_est, period=252, title='', params_iv={}, 
        plot_drawdown=True, plot_roll_sharpe=True, x_axis=None, 
        underlying='Underlying', win=252,
    ):
    """ 
    Print dynamic plot of performance indicators (perf, rolling sharpe and draw 
    down) of a strategy (raw and iso-volatility) versus its underlying.
    
    Parameters
    ----------
    :y_idx: np.ndarray[np.float64, ndim=1]
        Time series of log-returns of the underlying.
    :y_est: np.ndarray[np.float64, ndim=1]
        Time series of the signal's strategy.
    :period: int
        Number of period per year.
    :title: str or list of str
        Title of performance strategy.
    :plot_drawdown: bool (default is True)
        If true plot drawdowns.
    :plot_roll_sharpe: bool (default is True)
        If true plot rolling sharpe ratios.
    :x_axis: list or np.asarray
        x-axis to plot (e.g. list of dates).
    :underlying: str (default is 'Underlying')
        Name of the underlying.
    :win: int (default is 252)
        Size of the window of rolling sharpe.

    Returns
    -------
    :perf_idx: np.ndarray[np.float64, ndim=1]
        Time series of underlying performance.
    :perf_est: np.ndarray[np.float64, ndim=1]
        Time series of raw strategy performance.
    :perf_ivo: np.ndarray[np.float64, ndim=1]
        Time series of iso-vol strategy performance.
    """
    if x_axis is None:
        x_axis=range(y_idx.size)
    # Compute perf. 
    perf_idx = np.exp(np.cumsum(y_idx))
    perf_est = np.exp(np.cumsum(y_idx * y_est))
    iv = iso_vol(np.exp(np.cumsum(y_idx)), **params_iv) 
    perf_ivo = np.exp(np.cumsum(y_idx * y_est * iv))

    # Print stats. table 
    txt = set_text_stats(
        y_idx, period=period,
        Strategy=y_est,
        Strat_IsoVol=y_est * iv,
        underlying=underlying
    )
    print(txt)

    # Plot results
    n = 1 + plot_roll_sharpe + plot_drawdown
    f, ax = plt.subplots(n, 1, figsize=(9, 6), sharex=True)
    if n == 1:
        ax_perf, ax_dd, ax_dd = ax, None, None
        ax_perf.set_xlabel('Date')
    elif n == 2: 
        ax_perf = ax[0]
        ax_dd, ax_roll = ax[1], None if plot_drawdown else None, ax[1]
        ax[-1].set_xlabel('Date')
    else:
        ax_perf, ax_dd, ax_roll = ax[0], ax[1], ax[2]
    # Plot performances
    l_idx, = ax_perf.plot(
        x_axis,
        100 * perf_est, 
        color=sns.xkcd_rgb["pale red"], 
        LineWidth=2.
    )
    l_str, = ax_perf.plot(
        x_axis,
        100 * perf_ivo, 
        color=sns.xkcd_rgb["medium green"], 
        LineWidth=1.8
    )
    l_siv, = ax_perf.plot(
        x_axis,
        100 * perf_idx, 
        color=sns.xkcd_rgb["denim blue"], 
        LineWidth=1.5
    )

    # Set notify motion function
    def motion(event):
        N = len(ax_perf.lines[0].get_ydata())
        w, h = f.get_size_inches() * f.dpi - 200
        x = max(event.x - 100, 0)
        j = int(x / w * N)
        ax_perf.legend([
            'Strategy: {:.0f} %'.format(ax_perf.lines[0].get_ydata()[j] - 100),
            'Strat Iso-Vol: {:.0f} %'.format(ax_perf.lines[1].get_ydata()[j] - 100),
            '{}: {:.0f} %'.format(underlying, ax_perf.lines[2].get_ydata()[j] - 100),
        ], loc='upper left', frameon=True, fontsize=10)
        if plot_drawdown is not None:
            ax_dd.legend([
                'Strategy: {:.2f} %'.format(ax_dd.lines[0].get_ydata()[j]),
                'Strat Iso-Vol: {:.2f} %'.format(ax_dd.lines[1].get_ydata()[j]),
                '{}: {:.2f} %'.format(underlying, ax_dd.lines[2].get_ydata()[j]),
            ], loc='upper left', frameon=True, fontsize=10)
        if plot_roll_sharpe is not None:
            ax_roll.legend([
                'Strategy: {:.2f}'.format(ax_roll.lines[0].get_ydata()[j]),
                'Strat Iso-Vol: {:.2f}'.format(ax_roll.lines[1].get_ydata()[j]),
                '{}: {:.2f}'.format(underlying, ax_roll.lines[2].get_ydata()[j]),
            ], loc='upper left', frameon=True, fontsize=10)

    ax_perf.legend(
        ['Strategy', 'Strat Iso-Vol', underlying],
        loc='upper left', frameon=True, fontsize=10
    )
    ax_perf.set_ylabel('Perf.')
    ax_perf.set_yscale('log')
    ax_perf.set_title(title)
    ax_perf.tick_params(axis='x', rotation=30, labelsize=10)
    # Plot DrawDowns
    if plot_drawdown is not None:
        ax_dd.plot(
            x_axis,
            100 * drawdown(perf_est), 
            color=sns.xkcd_rgb["pale red"], 
            LineWidth=1.4
        )
        ax_dd.plot(
            x_axis,
            100 * drawdown(perf_ivo), 
            color=sns.xkcd_rgb["medium green"], 
            LineWidth=1.2
        )
        ax_dd.plot(
            x_axis,
            100 * drawdown(perf_idx), 
            color=sns.xkcd_rgb["denim blue"], 
            LineWidth=1.
        )
        ax_dd.set_ylabel('% DrawDown')
        ax_dd.set_title('DrawDown in percentage')
        ax_dd.tick_params(axis='x', rotation=30, labelsize=10)
    # Plot rolling Sharpe ratio
    if plot_roll_sharpe is not None:
        ax_roll.plot(
            x_axis,
            roll_sharpe(perf_est, period=period, win=win), 
            color=sns.xkcd_rgb["pale red"], 
            LineWidth=1.4
        )
        ax_roll.plot(
            x_axis,
            roll_sharpe(perf_ivo, period=period, win=win), 
            color=sns.xkcd_rgb["medium green"], 
            LineWidth=1.2
        )
        ax_roll.plot(
            x_axis,
            roll_sharpe(perf_idx, period=period, win=win), 
            color=sns.xkcd_rgb["denim blue"], 
            LineWidth=1.
        )
        ax_roll.set_ylabel('Sharpe ratio')
        ax_roll.set_xlabel('Date')
        ax_roll.set_title('Rolling Sharpe ratio')
        ax_roll.tick_params(axis='x', rotation=30, labelsize=10)

    cid = f.canvas.mpl_connect('motion_notify_event', motion)
    plt.show()

    return perf_idx, perf_est, perf_ivo