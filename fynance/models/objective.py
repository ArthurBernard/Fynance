#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Objective-aligned training.

:class:`ObjectiveModel` trains a neural network **directly on a risk-adjusted
objective** (e.g. :class:`~fynance.models.SharpeLoss`) instead of MSE against a
target: the network outputs **positions**, and the loss is computed on the
strategy returns ``positions * returns``. It conforms to the ``SignalModel``
protocol (``fit``/``predict``), so it drops into the harness via the precomputed
``X`` path::

    from fynance.models import ObjectiveModel, SharpeLoss
    from fynance.strategy import Strategy

    model = ObjectiveModel(loss=SharpeLoss(), epochs=80)
    strat = Strategy(model=model, signal=lambda p: p)   # net already outputs positions
    run_experiment(strat, prices, X=features, y=returns, walk_forward=...)

``fit(X, y)`` interprets ``y`` as the **realized per-bar returns** aligned with
``X``; ``predict(X)`` returns positions in ``[-1, 1]`` (via ``tanh``).

"""

# Built-in
from __future__ import annotations

from typing import Any, Callable

# Third-party
import numpy as np
import torch
from numpy.typing import NDArray

# Local
from fynance.models.loss import SharpeLoss

__all__ = ['ObjectiveModel']


def _default_net(n_features: int, layers: tuple[int, ...]) -> torch.nn.Module:
    """ A plain feed-forward net with ReLU hidden layers and a linear head. """
    mods: list[torch.nn.Module] = []
    dim = n_features
    for h in layers:
        mods += [torch.nn.Linear(dim, h), torch.nn.ReLU()]
        dim = h
    mods += [torch.nn.Linear(dim, 1)]  # linear position head

    return torch.nn.Sequential(*mods)


class ObjectiveModel:
    """ Train a net to maximize a differentiable financial objective.

    Parameters
    ----------
    net : torch.nn.Module, optional
        Architecture mapping a feature matrix ``(T, F)`` to ``(T, 1)``. Defaults
        to an MLP built lazily on the first :meth:`fit` (so it learns ``F``). Pass
        any ``nn.Module`` (e.g. a TCN/LSTM) to use a custom architecture.
    layers : tuple of int
        Hidden sizes of the default MLP (ignored when ``net`` is given).
    loss : BaseLoss, optional
        Differentiable financial loss applied to the strategy returns
        ``positions * returns``. Defaults to :class:`SharpeLoss`.
    optimizer : type[torch.optim.Optimizer]
        Optimizer class (default :class:`~torch.optim.Adam`).
    lr : float
        Learning rate.
    epochs : int
        Passes over the data per :meth:`fit`. With full-batch (``batch_size``
        ``None``) this is the number of optimizer steps; with mini-batches it is
        ``epochs * ceil(T / batch_size)`` steps — **far more updates**, which the
        objective usually needs to converge on long series.
    batch_size : int, optional
        Train on **contiguous** mini-batches of this many bars (order preserved so
        the turnover penalty stays meaningful). ``None`` (default) = full batch.
        Mini-batching is the practical way to actually train on long (e.g. minute)
        series — full-batch gives only ``epochs`` gradient steps total.
    shuffle : bool
        When mini-batching, shuffle the **order of the contiguous chunks** each
        epoch (rows within a chunk stay ordered). Improves SGD; default True.
    position_fn : callable
        Maps the net output to a position; default ``tanh`` (positions in
        ``[-1, 1]``).
    cost : float
        Per-bar proportional turnover cost penalized **during training** (e.g.
        ``0.0026`` for 26 bps). When non-zero the objective is computed on the
        **net-of-cost** return ``positions * returns - cost * |Δpositions|``, so
        the net learns to hold positions instead of churning — the anti-churn
        brick for high-cost / high-frequency settings. Use the same value as the
        backtest's :class:`~fynance.backtest.ProportionalCost`. Default ``0``
        (no penalty, original behaviour).
    seed : int
        Seed for reproducible initialization/training.

    Notes
    -----
    The net is **warm-started** across successive :meth:`fit` calls (so a
    walk-forward refit adapts online). Build a fresh model for an independent run.

    """

    def __init__(
        self,
        net: torch.nn.Module | None = None,
        *,
        layers: tuple[int, ...] = (16, 8),
        loss: Any = None,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        epochs: int = 80,
        batch_size: int | None = None,
        shuffle: bool = True,
        position_fn: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        cost: float = 0.0,
        seed: int = 0,
    ):
        self.net = net
        self.layers = tuple(layers)
        self.loss = loss if loss is not None else SharpeLoss()
        self.optimizer_cls = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.position_fn = position_fn
        self.cost = cost
        self.seed = seed
        self._optim: torch.optim.Optimizer | None = None

    def _ensure_net(self, n_features: int) -> None:
        if self.net is None:
            torch.manual_seed(self.seed)
            self.net = _default_net(n_features, self.layers)
        if self._optim is None:
            self._optim = self.optimizer_cls(
                self.net.parameters(), lr=self.lr,  # type: ignore[call-arg]
            )

    def _positions(self, X: torch.Tensor) -> torch.Tensor:
        out = self.net(X).reshape(-1)  # type: ignore[misc]

        return self.position_fn(out)

    def _strat_return(self, pos: torch.Tensor, ret: torch.Tensor,
                      prev: torch.Tensor | None) -> torch.Tensor:
        """ Net-of-cost strategy return ``pos*ret - cost*|Δpos|`` for a chunk.

        ``prev`` is the (detached) last position of the previous contiguous chunk
        so the turnover at the chunk boundary is charged correctly; ``None`` (or a
        shuffled chunk) charges entry from flat on the first bar.
        """
        strat = pos * ret
        if self.cost:
            first = pos[:1].abs() if prev is None else (pos[:1] - prev).abs()
            turnover = torch.cat([first, torch.abs(pos[1:] - pos[:-1])])
            strat = strat - self.cost * turnover

        return strat

    def fit(self, X: NDArray, y: NDArray) -> ObjectiveModel:
        """ Train the net to maximize the objective of the net-of-cost return.

        Parameters
        ----------
        X : array-like, shape (T, F)
            Feature matrix.
        y : array-like, shape (T,)
            Realized per-bar returns aligned with ``X`` (not a supervised label).

        Returns
        -------
        ObjectiveModel
            ``self``.

        """
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        rt = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1))
        self._ensure_net(Xt.shape[1])

        T = Xt.shape[0]
        bs = self.batch_size or T
        n_chunks = (T + bs - 1) // bs
        gen = torch.Generator().manual_seed(self.seed)

        self.net.train()  # type: ignore[union-attr]
        for _ in range(self.epochs):
            order: list[int] = list(range(n_chunks))
            if self.shuffle and n_chunks > 1:
                order = torch.randperm(n_chunks, generator=gen).tolist()

            prev: torch.Tensor | None = None
            for ci in order:
                a, b = ci * bs, min((ci + 1) * bs, T)
                self._optim.zero_grad()  # type: ignore[union-attr]
                pos = self._positions(Xt[a:b])
                # Carry the previous chunk's last position only when chunks run in
                # time order (no shuffle); a shuffled chunk charges entry-from-flat.
                strat_ret = self._strat_return(pos, rt[a:b],
                                               None if self.shuffle else prev)
                loss = self.loss(strat_ret)
                loss.backward()
                self._optim.step()  # type: ignore[union-attr]
                prev = pos[-1].detach()

        return self

    @torch.no_grad()
    def predict(self, X: NDArray) -> NDArray:
        """ Return positions in ``[-1, 1]`` for each row of ``X``. """
        self.net.eval()  # type: ignore[union-attr]
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))

        return self._positions(Xt).reshape(-1, 1).cpu().numpy()
