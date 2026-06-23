#!/usr/bin/env python3
# coding: utf-8

""" Optional orchestrator composing the fynance pipeline.

:class:`Strategy` wires the protocol-based maillons — features -> model ->
signal -> cost -> backtest -> metrics — into one runnable object. Every slot is
optional and accepts any object conforming to its protocol, so the pieces stay
usable standalone (this is composition, not a forced pipeline).

Decision D4: a fluent dataclass-style constructor (slots as keyword arguments)
with a ``.run`` method, rather than a declarative config tree.

"""

from __future__ import annotations

# Built-in packages
from typing import Any, Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.backtest import backtest
from fynance.backtest.result import BacktestResult
from fynance.data import walk_forward
from fynance.signal import sign

__all__ = ['Strategy']


def _as_array(data: Any) -> NDArray:
    """ Coerce a PriceSeries / array-like to a float64 numpy array. """
    values = getattr(data, "values", None)
    arr = values if values is not None else data

    return np.asarray(arr, dtype=np.float64)


class Strategy:
    """ Compose features, a model, a signal mapper and costs into a backtest.

    Parameters
    ----------
    model : SignalModel, optional
        Object with ``fit(X, y)`` / ``predict(X)``. If omitted, the (featured)
        input is fed straight to the signal mapper (rule-based strategy).
    signal : callable, optional
        Maps predictions/features to positions. Defaults to
        :func:`fynance.signal.sign`.
    features : callable, optional
        Maps the price series to a feature array used as model/​signal input.
        Defaults to the identity (the price series itself).
    cost : CostModel, optional
        Per-step transaction cost model.
    period : int
        Annualization factor passed to the summary.

    """

    def __init__(
        self,
        model: Any = None,
        signal: Callable[..., NDArray] = sign,
        features: Callable[[NDArray], NDArray] | None = None,
        cost: Any = None,
        period: int = 252,
    ):
        """ Store the pipeline slots. """
        self.model = model
        self.signal = signal
        self.features = features
        self.cost = cost
        self.period = period

    def _featurize(self, prices: NDArray) -> NDArray:
        """ Build the feature array from a price series. """
        if self.features is None:

            return prices

        return np.asarray(self.features(prices), dtype=np.float64)

    def _resolve_features(self, prices: NDArray, X: NDArray | None) -> NDArray:
        """ Use a precomputed feature matrix ``X`` if given, else featurize prices.

        ``X`` is the caller's responsibility to keep **aligned with the price
        index** and **causal** — it carries exogenous features (engineered inputs,
        a regime label, other venues) that the price-only featurizer cannot build.
        Its dtype is preserved (so a float32 ``X`` matches a float32 torch model).
        """
        if X is not None:

            return np.asarray(X)

        return self._featurize(prices)

    def _positions(self, prices: NDArray, y: NDArray | None,
                   X: NDArray | None = None) -> NDArray:
        """ Compute the (unshifted) position series. """
        feats = self._resolve_features(prices, X)

        if self.model is not None:
            if y is not None:
                self.model.fit(feats, y)

            preds = np.asarray(self.model.predict(feats), dtype=np.float64)

        else:
            preds = feats

        return self._signal_positions(preds)

    def _signal_positions(self, preds: NDArray) -> NDArray:
        """ Map predictions to positions, preserving a ``(T, N)`` book.

        A genuine multi-asset book ``(T, N)`` is kept 2-D; only a trailing
        singleton asset axis (the single-asset model output ``(T, 1)``) is
        collapsed to 1-D so the single-asset path is unchanged.
        """
        pos = np.asarray(self.signal(preds), dtype=np.float64)
        if pos.ndim == 2 and pos.shape[1] == 1:
            pos = pos.reshape(-1)

        return pos

    def run(self, data: Any, y: NDArray | None = None,
            X: NDArray | None = None) -> BacktestResult:
        """ Run the strategy on a price series, returning a backtest result.

        Parameters
        ----------
        data : PriceSeries or array-like
            Price series (used for the P&L).
        y : array-like, optional
            Supervised target for the model (fit on the whole series; for
            leak-free training use :meth:`run_walk_forward`).
        X : array-like, optional
            Precomputed feature matrix aligned with ``data`` (rows = time). When
            given it replaces ``features(prices)`` — use it to feed exogenous /
            regime / multi-venue inputs the price-only featurizer cannot build.

        Returns
        -------
        BacktestResult

        """
        prices = _as_array(data)
        returns = prices[1:] / prices[:-1] - 1.0
        positions = self._positions(prices, y, X)

        # Align positions with the returns series (drop the first observation).
        if positions.shape[0] == prices.shape[0]:
            positions = positions[1:]

        return backtest(returns, positions, cost=self.cost, shift=True)

    def run_walk_forward(
        self,
        data: Any,
        y: NDArray,
        train: int,
        test: int,
        step: int | None = None,
        purge: int = 0,
        X: NDArray | None = None,
    ) -> BacktestResult:
        """ Walk-forward run: refit per window on train only, predict on test.

        The model and features are fit on each train slice only; out-of-sample
        positions are stitched together and backtested. Strictly no-lookahead.
        This per-window refit *is* the rolling-NN pattern when ``model`` is a net.

        .. note::
           The **first realized return of every test block is discarded** (set
           to ``0``): within a block the return at bar ``k`` is computed from
           prices ``k-1`` and ``k``, so the opening bar has no in-block prior
           and earns nothing. With the default ``step == test`` this drops one
           bar per block (roughly ``1 / test`` of the out-of-sample sample);
           the dropped return is the gap return across the block join, which
           would require carrying the previous block's closing position to
           realize causally. This conservative choice avoids any cross-block
           lookahead at the cost of a slightly truncated OOS sample.

        Parameters
        ----------
        data : PriceSeries or array-like
            Price series (used for the P&L).
        y : array-like
            Supervised target aligned with ``data``.
        train, test : int
            Train and test window lengths.
        step : int, optional
            Roll step (defaults to ``test``).
        purge : int
            Embargo removed at the train/test boundary.
        X : array-like, optional
            Precomputed feature matrix aligned with ``data`` (rows = time). When
            given, each window slices ``X[train]`` / ``X[test]`` instead of
            featurizing the price slices — the way to feed exogenous / regime /
            multi-venue features. ``X`` must be causal and index-aligned.

        Returns
        -------
        BacktestResult
            Backtest of the concatenated out-of-sample positions.

        """
        prices = _as_array(data)
        y_arr = np.asarray(y)  # preserve caller dtype (match X / the model)
        X_arr = None if X is None else np.asarray(X)  # preserve caller dtype
        n = prices.shape[0]

        oos_pos: list[NDArray] = []
        oos_ret: list[NDArray] = []

        for tr, te in walk_forward(n, train=train, test=test, step=step,
                                   purge=purge):
            if X_arr is not None:
                feats_tr = X_arr[tr]
                feats_te = X_arr[te]

            else:
                feats_tr = self._featurize(prices[tr])
                feats_te = self._featurize(prices[te])

            if self.model is not None:
                self.model.fit(feats_tr, y_arr[tr])
                preds = np.asarray(self.model.predict(feats_te), dtype=np.float64)

            else:
                preds = feats_te

            pos = self._signal_positions(preds)

            # Return realized over each test step (causal: position at t earns
            # r_t within the OOS block; the engine applies the one-step shift).
            # The opening bar has no in-block prior price, so its return is
            # discarded (NaN -> 0 below). For a panel the whole opening row is
            # discarded. See the method docstring note.
            block = prices[te]
            ret = np.empty_like(block, dtype=np.float64)
            ret[0] = np.nan
            ret[1:] = block[1:] / block[:-1] - 1.0

            oos_pos.append(pos)
            oos_ret.append(ret)

        # Stitch the out-of-sample blocks: 1-D for a single asset, (T_oos, N)
        # for a panel book.
        positions = np.concatenate(oos_pos, axis=0)
        returns = np.nan_to_num(np.concatenate(oos_ret, axis=0), nan=0.0)

        return backtest(returns, positions, cost=self.cost, shift=True)
