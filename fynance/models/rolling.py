#!/usr/bin/env python3
# coding: utf-8

""" Walk-forward training wrappers for time-series models.

Iterator-based base class :class:`_RollingBasis` that re-fits a model
on a sliding training window and predicts on the next out-of-sample
window. The pattern enforces strict temporal ordering, eliminating
lookahead bias and matching how a strategy would actually be retrained
in production.

A concrete :class:`RollMultiLayerPerceptron` combines this iterator
with :class:`fynance.models.neural_network.MultiLayerPerceptron`. The
same pattern is applied to portfolio allocation in
:func:`fynance.algorithms.allocation.rolling_allocation`.

Main entry points
-----------------
- :class:`_RollingBasis` — iterator that yields ``(eval_set,
  test_set)`` slices and tracks training/evaluation/test losses.
- :class:`RollMultiLayerPerceptron` — walk-forward MLP, ready to use
  via :meth:`RollMultiLayerPerceptron.set_roll_period` and
  :meth:`RollMultiLayerPerceptron.run`.

"""

from __future__ import annotations

# Built-in packages
from multiprocessing import Process
from typing import Callable

# External packages
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from fynance.backtest.dynamic_plot_backtest import BacktestNeuralNet

# Local packages
from fynance.models.neural_network import MultiLayerPerceptron

plt.style.use('seaborn-v0_8')


__all__ = ['_RollingBasis', 'RollMultiLayerPerceptron']


class _RollingBasis:
    r""" Base object to roll a model over a time axis.

    At each step the model trains on ``X[t-n:t]`` and predicts on
    ``X[t:t+s]``.  Call :meth:`set_roll_period` (or :meth:`__call__`) to
    configure the window sizes, then iterate with :func:`run`.

    The leading underscore signals that this class is **internal** —
    its ``_train``, ``_get_loss_on`` and ``sub_predict`` hooks are
    expected to be overridden by a concrete subclass mixed with a
    :class:`~fynance.models.neural_network.BaseNeuralNet` descendant.
    The public, stable entry point is
    :class:`RollMultiLayerPerceptron`. Other ready-to-use combinations
    can be added by following the same pattern (multiple inheritance
    + ``set_roll_period`` instead of ``__call__``, since the latter is
    captured by ``torch.nn.Module``).

    Parameters
    ----------
    X, y : array_like
        Respectively input and output data, shaped ``(T, N)`` and
        ``(T, M)``. Strict temporal ordering is required — no
        shuffling, no future leakage.
    f : callable, optional
        Function to transform target, e.g. ``torch.sign``.
    index : array_like, optional
        Time index of data.

    Attributes
    ----------
    n, s, r : int
        Respectively size of training, testing and rolling period.
    b, e, T : int
        Respectively batch size, number of epochs and size of entire dataset.
    t, _e, i : int
        Respectively the current time period, the current epoch and the
        current iteration.
    n_iter : int
        Total number of iterations.
    y_eval, y_test : np.ndarray
        Respectively evaluating and testing predictions.
    log : list of dict
        Per-step record of ``{step, train_loss, eval_loss, test_loss}``,
        populated by :meth:`run`.  Use :meth:`get_stats` to get a DataFrame.

    """

    def __init__(
        self,
        X: NDArray | torch.Tensor,
        y: NDArray | torch.Tensor,
        f: Callable | None = None,
        index: NDArray | None = None,
    ):
        self.T = X.shape[0]
        self.y_shape = y.shape
        self.f = (lambda x: x) if f is None else f
        self.idx = np.arange(self.T) if index is None else index
        self.log: list[str] = []

    def __call__(
        self,
        train_period: int,
        test_period: int,
        start: int = 0,
        end: int | None = None,
        roll_period: int | None = None,
        eval_period: int | None = None,
        batch_size: int = 64,
        epochs: int = 1,
    ) -> _RollingBasis:
        """ Configure rolling window parameters.

        Parameters
        ----------
        train_period, test_period : int
            Size of respectively training and testing sub-periods.
        start : int, optional
            Starting observation, default is first observation.
        end : int, optional
            Ending observation, default is last observation.
        roll_period : int, optional
            Size of the rolling period, default equals ``test_period``.
        eval_period : int, optional
            Size of the evaluating period (unused, kept for API compat).
        batch_size : int, optional
            Training batch size, default is 64.
        epochs : int, optional
            Number of epochs per sub-period, default is 1.

        Returns
        -------
        _RollingBasis

        """
        self.n = train_period
        self.s = test_period
        self.r = test_period if roll_period is None else roll_period
        self.b = batch_size
        self.e = epochs

        self.T = self.T if end is None else min(self.T, end)
        self.t0 = max(self.n - self.r, min(start, self.T - self.n - self.s))
        self.n_iter = (self.T - self.t0 - self.s) // self.r * self.e
        self.log = []

        return self

    def __iter__(self):
        self.y_eval = np.zeros(self.y_shape, dtype=np.float64)
        self.y_test = np.zeros(self.y_shape, dtype=np.float64)
        self.loss_eval = np.zeros([self.n_iter], dtype=np.float64)
        self.loss_test = np.zeros([self.n_iter], dtype=np.float64)
        self.loss_train = np.zeros([self.n_iter], dtype=np.float64)
        self._e = self.e
        self.t = self.t0
        self.i = -1

        return self

    def __next__(self):
        self._e += 1
        self.i += 1
        if self._e > self.e:
            self._e = 1
            if self.t + self.r + self.s > self.T:
                raise StopIteration
            self.t += self.r
            self.t_idx = np.arange(self.t - self.n, self.t)

        eval_set = slice(self.t - self.r, self.t)
        test_set = slice(self.t, self.t + self.s)

        return eval_set, test_set

    def get_stats(self):
        """ Return per-step loss history as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``train_loss``, ``eval_loss``, ``test_loss``.

        """
        if not self.log:
            return pd.DataFrame(
                columns=['step', 'train_loss', 'eval_loss', 'test_loss']
            )
        return pd.DataFrame(self.log)

    def plot_loss(self, figsize=(9, 4)):
        """ Plot train / eval / test loss curves.

        Parameters
        ----------
        figsize : tuple of int, optional

        Returns
        -------
        matplotlib.figure.Figure

        """
        df = self.get_stats()
        if df.empty:
            raise RuntimeError('No log data — run the model first.')

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df['step'], df['train_loss'], label='Train')
        ax.plot(df['step'], df['eval_loss'], label='Eval')
        ax.plot(df['step'], df['test_loss'], label='Test')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.tight_layout()

        return fig

    def _training(self):
        loss_epoch = 0.
        np.random.shuffle(self.t_idx)
        for t in range(0, self.n, self.b):
            s = min(t + self.b, self.n)
            train_slice = self.t_idx[t: s]
            try:
                lo = self._train(
                    X=self.X[train_slice],
                    y=self.f(self.y[train_slice]),
                )
            except Exception as e:
                print(train_slice)
                print(self.X[train_slice])
                print(self.f(self.y[train_slice]))
                raise e
            loss_epoch += lo.item()

        self.loss_train[self.i] = loss_epoch / s

    def run(self, backtest_plot=True, backtest_kpi=True, figsize=(9, 6),
            func=np.sign):
        """ Run the rolling model and collect backtest predictions.

        Parameters
        ----------
        backtest_plot : bool, optional
            If True, display a live backtest performance plot.
        backtest_kpi : bool, optional
            If True, print KPIs to stdout at each step.
        figsize : tuple of int, optional
            Figure size.
        func : callable, optional
            Function applied to predictions before computing returns.

        """
        y = self.y.numpy()
        r = np.exp(y) - 1
        y_perf = np.exp(np.cumsum(y, axis=0))
        y_perf = 100. * y_perf / y_perf[self.t0]
        self.perf_eval = 100. * np.ones(y.shape, dtype=np.float64)
        self.perf_test = 100. * np.ones(y.shape, dtype=np.float64)

        self.bnn = BacktestNeuralNet(figsize)
        self.log = []
        p_print = None

        for eval_set, test_set in self:
            self._training()

            self.y_eval[eval_set] = self.sub_predict(self.X[eval_set])
            self.y_test[test_set] = self.sub_predict(self.X[test_set])
            self.loss_eval[self.i] = self._get_loss_on(self.y_eval, eval_set)
            self.loss_test[self.i] = self._get_loss_on(self.y_test, test_set)

            self.log.append({
                'step': self.i,
                'train_loss': self.loss_train[self.i],
                'eval_loss': self.loss_eval[self.i],
                'test_loss': self.loss_test[self.i],
            })

            if self._e == self.e:
                v0 = self.perf_eval[self.t - self.r - 1]
                self.perf_eval[eval_set] = get_perf2(
                    r[eval_set], func(self.y_eval[eval_set]), v0=v0
                )
                v0 = self.perf_test[self.t - 1]
                self.perf_test[test_set] = get_perf2(
                    r[test_set], func(self.y_test[test_set]), v0=v0
                )

            if self.t > self.t0 + self.r:
                if p_print is None or not p_print.is_alive():
                    p_print = Process(
                        target=self._print,
                        args=(self.t, self.i, r, y_perf, func,
                              backtest_plot, backtest_kpi)
                    )
                    p_print.start()

        self._print(self.t, self.i, r, y_perf, func, backtest_plot,
                    backtest_kpi)

        return self

    def _print(self, t, i, r, y_perf, func, backtest_plot, backtest_kpi):
        if backtest_kpi:
            self._display_kpi(t)

        if backtest_plot:
            self._display_plot_loss(self.bnn, i)
            self._display_plot_perf(
                self.bnn, self.perf_test, self.perf_eval, y_perf, t
            )
            self.bnn.f.canvas.draw()

    def _get_loss_on(self, y, _slice):
        lo = self.criterion(
            torch.from_numpy(y[_slice]).to(torch.float32),
            self.y[_slice].to(torch.float32)
        )
        return lo.item()

    def _display_kpi(self, t):
        pct = t - self.n - self.s
        pct = pct / (self.T - self.n - self.T % self.s)
        txt = '{:5.2%} is done | '.format(pct)
        txt += 'Eval loss is {:5.2} | '.format(self.loss_eval[-1])
        txt += 'Test loss is {:5.2} | '.format(self.loss_test[-1])
        print(txt, end='\r')

    def _display_plot_loss(self, bnn, i):
        bnn.plot_loss(self.loss_test[: i],
                      self.loss_eval[: i],
                      self.loss_train[: i])

    def _display_plot_perf(self, bnn, perf_test, perf_eval, y_perf, t):
        bnn.plot_perf(perf_test[self.t0: t + self.s],
                      perf_eval[self.t0 - self.s: t],
                      y_perf[self.t0 - self.s: t],
                      self.idx[self.t0 - self.s: t + self.s])


def get_perf2(ret, signal, v0=100):
    return v0 * np.cumprod(ret * signal + 1, axis=0)


def get_perf(signal, underlying, v0=100):
    return v0 * np.exp(np.cumsum(signal * underlying, axis=0))


class RollMultiLayerPerceptron(MultiLayerPerceptron, _RollingBasis):
    """ Rolling version of the multi-layer perceptron model.

    End-to-end walk-forward training pipeline for an MLP: at each step
    the model is fitted on a sliding window of length ``n``, evaluated
    on the previous out-of-sample slice, then used to predict the next
    slice. Losses (train, eval, test) and out-of-sample predictions are
    accumulated step by step in ``self.log``, ``self.y_eval`` and
    ``self.y_test`` for downstream analysis.

    Use :meth:`set_roll_period` to configure window sizes and batch
    options, then :meth:`run` to execute the loop. ``run`` can also
    drive a live :class:`fynance.backtest.dynamic_plot_backtest.BacktestNeuralNet`
    figure to monitor convergence.

    Combines :class:`MultiLayerPerceptron` with the walk-forward iterator
    from :class:`_RollingBasis`.  Use :meth:`set_roll_period` instead of
    calling the object directly (``__call__`` is captured by
    ``torch.nn.Module``).

    Methods
    -------
    set_roll_period
    sub_predict

    """

    def __init__(
        self,
        X: NDArray | torch.Tensor,
        y: NDArray | torch.Tensor,
        layers: list[int] = [],
        activation: type[torch.nn.Module] | None = None,
        drop: float | None = None,
        bias: bool = True,
        x_type=None,
        y_type=None,
        activation_kwargs: dict = {},
        **kwargs,
    ):
        _RollingBasis.__init__(self, X, y, **kwargs)
        MultiLayerPerceptron.__init__(self, X, y, layers=layers, bias=bias,
                                      activation=activation, drop=drop,
                                      x_type=x_type, y_type=y_type,
                                      activation_kwargs=activation_kwargs)

    def set_roll_period(
        self,
        train_period: int,
        test_period: int,
        start: int = 0,
        end: int | None = None,
        roll_period: int | None = None,
        eval_period: int | None = None,
        batch_size: int = 64,
        epochs: int = 1,
    ) -> _RollingBasis:
        """ Configure rolling window parameters.

        This is the preferred entry-point for ``RollMultiLayerPerceptron``
        because ``__call__`` is captured by ``torch.nn.Module``.

        Parameters
        ----------
        train_period, test_period : int
            Size of respectively training and testing sub-periods.
        start : int, optional
        end : int, optional
        roll_period : int, optional
        eval_period : int, optional
        batch_size : int, optional
        epochs : int, optional

        Returns
        -------
        _RollingBasis

        """
        return _RollingBasis.__call__(
            self, train_period=train_period, test_period=test_period,
            start=start, end=end, roll_period=roll_period,
            eval_period=eval_period, batch_size=batch_size, epochs=epochs
        )

    def _train(self, X, y):
        return self.train_on(X=X, y=y)

    def sub_predict(self, X: torch.Tensor) -> NDArray:
        """ Return predictions as a numpy array. """
        return self.predict(X=X).numpy()

    def save(self, path):
        """ Save the model weights.

        Parameters
        ----------
        path : str
            Destination path.

        """
        torch.save(self.state_dict(), path)
