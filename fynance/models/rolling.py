#!/usr/bin/env python3
# coding: utf-8

""" Walk-forward training wrappers for time-series models.

Iterator-based base class :class:`_RollingBasis` that re-fits a model
on a sliding training window and predicts on the next out-of-sample
window. The pattern enforces strict temporal ordering, eliminating
lookahead bias and matching how a strategy would actually be retrained
in production.

Each step trains on ``X[t-n:t]`` and yields two slices: ``eval_set =
slice(t-r, t)`` — the last ``r`` bars of the **training** window, used
for an in-sample fit-quality check — and ``test_set = slice(t, t+s)`` —
the next strictly out-of-sample window.

A concrete :class:`RollMultiLayerPerceptron` combines this iterator
with :class:`fynance.models.mlp.MultiLayerPerceptron`. The
same pattern is applied to portfolio allocation in
:func:`fynance.portfolio.allocation.rolling_allocation`.

Main entry points
-----------------
- :class:`_RollingBasis` — iterator that yields ``(eval_set,
  test_set)`` slices (in-sample tail + out-of-sample window) and tracks
  training/evaluation/test losses.
- :class:`RollMultiLayerPerceptron` — walk-forward MLP, ready to use
  via :meth:`RollMultiLayerPerceptron.set_roll_period` and
  :meth:`RollMultiLayerPerceptron.run`.

"""

from __future__ import annotations

# Built-in packages
from typing import Callable

# External packages
import numpy as np
import torch
from numpy.typing import NDArray

from fynance.backtest.backtest_neural_net import BacktestNeuralNet

# Local packages
from fynance.models.cv_result import CVResult
from fynance.models.mlp import MultiLayerPerceptron

__all__ = ['CVResult', '_RollingBasis', 'RollMultiLayerPerceptron']


class _RollingBasis:
    r""" Base object to roll a model over a time axis.

    At each step the model trains on ``X[t-n:t]`` and predicts on
    ``X[t:t+s]``.  Call :meth:`set_roll_period` (or :meth:`__call__`) to
    configure the window sizes, then iterate with :func:`run`.

    The leading underscore signals that this class is **internal** —
    its ``_train``, ``_get_loss_on`` and ``sub_predict`` hooks are
    expected to be overridden by a concrete subclass mixed with a
    :class:`~fynance.models._base.BaseNeuralNet` descendant.
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
        populated by :meth:`run`.  Use :meth:`get_stats` for a structured array.

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
            Size of the rolling period, default equals ``test_period``. Must
            not exceed ``train_period`` — the in-sample ``eval_set`` is the
            last ``roll_period`` bars of the training window.
        eval_period : int, optional
            Size of the evaluating period (unused, kept for API compat).
        batch_size : int, optional
            Training batch size, default is 64.
        epochs : int, optional
            Number of epochs per sub-period, default is 1.

        Returns
        -------
        _RollingBasis

        Raises
        ------
        ValueError
            If ``roll_period`` exceeds ``train_period`` (would push the
            in-sample evaluation window outside the training window).

        """
        self.n = train_period
        self.s = test_period
        self.r = test_period if roll_period is None else roll_period
        self.b = batch_size
        self.e = epochs

        # Causality guards. The eval window is the last ``r`` bars of the
        # training window (``slice(t - r, t)``), so a roll larger than the
        # train length (``r > n``) would make the eval window reach outside
        # the in-sample window. Combined with a negative ``t0`` it also turns
        # the eval / train slices into negative indices that wrap to the tail
        # of the array — i.e. silently leak FUTURE bars into training and
        # evaluation. Reject ``r > n`` and clamp the window start to ``>= 0``.
        if self.r > self.n:
            raise ValueError(
                f"roll_period ({self.r}) must not exceed train_period "
                f"({self.n}): the evaluation window is the last roll_period "
                "bars of the training window, so r > n would reach outside "
                "the in-sample window (lookahead leak)."
            )

        self.T = self.T if end is None else min(self.T, end)
        self.t0 = max(0, self.n - self.r, min(start, self.T - self.n - self.s))
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

        # ``eval_set`` is the last ``r`` bars of the *training* window
        # (in-sample): an enforced ``r <= n`` keeps it inside ``[t - n, t)``.
        # ``test_set`` is the next, strictly out-of-sample window.
        eval_set = slice(self.t - self.r, self.t)
        test_set = slice(self.t, self.t + self.s)

        return eval_set, test_set

    def _fold_slices(self, purge: int = 0):
        """ Yield ``(train_slice, test_slice)`` for every walk-forward fold.

        Unlike :meth:`__next__`, this generator has no epoch loop and
        allocates nothing — it is a pure windowing helper shared by
        :meth:`cross_validate` and any future CV utilities.

        Parameters
        ----------
        purge : int, optional
            Number of training observations to drop at the train/test border
            (Lopez de Prado *purging*) to avoid leakage from overlapping
            feature windows. Default 0 (no purge).

        Yields
        ------
        train_slice : slice
            Window ``[t - n, t - purge)`` used for training.
        test_slice : slice
            Window ``[t, t + s)`` used for out-of-sample evaluation.

        """
        t = self.t0 + self.r
        while t + self.s <= self.T:
            yield slice(t - self.n, t - purge), slice(t, t + self.s)
            t += self.r

    def cross_validate(self, model_factory, X, y, metric_fn=None, epochs=1, purge=0):
        """ Walk-forward cross-validation with out-of-fold predictions.

        At each fold a **fresh** model is created via ``model_factory()``,
        trained on the rolling training window, and used to predict the
        next out-of-sample window.  Results are accumulated across all
        folds without any state leaking between them.

        Call :meth:`__call__` (or :meth:`set_roll_period` for
        ``RollMultiLayerPerceptron``) to configure ``train_period``,
        ``test_period``, and ``roll_period`` before calling this method.

        Parameters
        ----------
        model_factory : callable
            Called with no arguments before every fold.  Must return an
            object that exposes ``train_on(X, y)`` and
            ``predict(X) -> NDArray | Tensor`` (the
            :class:`~fynance.models._base.BaseNeuralNet`
            interface).
        X, y : array_like
            Input and target arrays shaped ``(T, N)`` and ``(T, M)``.
        metric_fn : callable, optional
            ``metric_fn(y_true, y_pred) -> float`` evaluated on the test
            window of each fold.  If None, :attr:`CVResult.fold_metrics`
            is an empty list and the mean/std fields are None.
        epochs : int, optional
            Number of full training passes per fold, default 1.
        purge : int, optional
            Training observations to drop at the train/test border (purging).
            Default 0.

        Returns
        -------
        CVResult

        Examples
        --------
        >>> import numpy as np
        >>> import torch, torch.nn as nn
        >>> from fynance.models.mlp import MultiLayerPerceptron
        >>> from fynance.models.rolling import _RollingBasis
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((80, 4)).astype(np.float32)
        >>> y = rng.standard_normal((80, 1)).astype(np.float32)
        >>> Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        >>> def factory():
        ...     m = MultiLayerPerceptron(4, 1, layers=[8])
        ...     m.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        ...     return m
        >>> rb = _RollingBasis(Xt, yt)
        >>> _ = rb(train_period=40, test_period=10, roll_period=10)
        >>> result = rb.cross_validate(factory, Xt, yt)
        >>> result.oof_predictions.shape
        (80, 1)

        """
        n_out = y.shape[1] if y.ndim > 1 else 1
        oof = np.full((X.shape[0], n_out), np.nan)
        fold_metrics = []

        for train_sl, test_sl in self._fold_slices(purge=purge):
            model = model_factory()
            X_tr, y_tr = X[train_sl], y[train_sl]
            for _ in range(epochs):
                model.train_on(X_tr, y_tr)
            pred = model.predict(X[test_sl])
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            oof[test_sl] = pred.reshape(-1, n_out)
            if metric_fn is not None:
                fold_metrics.append(float(metric_fn(y[test_sl], pred)))

        mean_m = float(np.mean(fold_metrics)) if fold_metrics else None
        std_m = float(np.std(fold_metrics)) if fold_metrics else None
        return CVResult(oof, fold_metrics, mean_m, std_m)

    def get_stats(self):
        """ Return per-step loss history as a structured array.

        Returns
        -------
        numpy.ndarray
            Structured array with fields ``step`` (int) and ``train_loss``,
            ``eval_loss``, ``test_loss`` (float). Access a column by name,
            e.g. ``stats["train_loss"]``; ``stats.size`` is the step count.

        """
        dtype = [('step', np.int64), ('train_loss', np.float64),
                 ('eval_loss', np.float64), ('test_loss', np.float64)]
        rows = [(r['step'], r['train_loss'], r['eval_loss'], r['test_loss'])
                for r in self.log]

        return np.array(rows, dtype=dtype)

    def plot_loss(self, figsize=(9, 4)):
        """ Plot train / eval / test loss curves.

        Parameters
        ----------
        figsize : tuple of int, optional

        Returns
        -------
        matplotlib.figure.Figure

        """
        from matplotlib import pyplot as plt  # lazy: keeps matplotlib off the
        # `import fynance` path (plotting is opt-in).
        plt.style.use('seaborn-v0_8')

        df = self.get_stats()
        if df.size == 0:
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
        n_batches = 0
        np.random.shuffle(self.t_idx)
        for t in range(0, self.n, self.b):
            s = min(t + self.b, self.n)
            train_slice = self.t_idx[t: s]
            lo = self._train(
                X=self.X[train_slice],
                y=self.f(self.y[train_slice]),
            )
            loss_epoch += lo.item()
            n_batches += 1

        # ``lo`` is already a per-batch *mean* loss, so the epoch loss is the
        # mean over batches — divide by the batch count, not by the train
        # length ``n`` (the stale ``s`` from the last batch) which made the
        # reported loss depend on the batch size and span a different scale
        # than a single-batch loss.
        self.loss_train[self.i] = loss_epoch / n_batches

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

            # Render in-process. A forked multiprocessing.Process is not
            # fork-safe here: it would share live torch tensors and the
            # matplotlib figure across the fork. Display is cheap and purely
            # a side effect, so do it inline.
            if self.t > self.t0 + self.r:
                self._print(self.t, self.i, r, y_perf, func,
                            backtest_plot, backtest_kpi)

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
        # Guard the denominator: ``T - n - T % s`` can be 0 (e.g. T=45, n=40,
        # s=10) which would raise ``ZeroDivisionError``. Floor it at 1.
        pct = pct / max(self.T - self.n - self.T % self.s, 1)
        txt = '{:5.2%} is done | '.format(pct)
        # Use the current iteration index ``self.i`` — ``[-1]`` is the last
        # array slot, which stays 0 until the final iteration. ``__next__``
        # bumps ``self.i`` before the ``StopIteration`` check, so after the
        # loop ``self.i == loss_eval.size``; clamp to the last filled slot to
        # avoid an out-of-bounds read on the final out-of-loop print.
        i = min(self.i, self.loss_eval.size - 1)
        txt += 'Eval loss is {:5.2} | '.format(self.loss_eval[i])
        txt += 'Test loss is {:5.2} | '.format(self.loss_test[i])
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


class RollMultiLayerPerceptron(MultiLayerPerceptron, _RollingBasis):
    """ Rolling version of the multi-layer perceptron model.

    End-to-end walk-forward training pipeline for an MLP: at each step
    the model is fitted on a sliding window of length ``n``, evaluated
    on the last ``r`` bars of that **training** window (the in-sample
    ``eval_set = slice(t-r, t)``, a fit-quality check — *not* out-of-sample
    data), then used to predict the next, strictly out-of-sample slice
    ``test_set = slice(t, t+s)``. Losses (train, eval, test) and
    out-of-sample predictions are accumulated step by step in ``self.log``,
    ``self.y_eval`` and ``self.y_test`` for downstream analysis.

    Use :meth:`set_roll_period` to configure window sizes and batch
    options, then :meth:`run` to execute the loop. ``run`` can also
    drive a live :class:`fynance.backtest.backtest_neural_net.BacktestNeuralNet`
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
