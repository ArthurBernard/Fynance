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
        Full-batch training steps per :meth:`fit`.
    position_fn : callable
        Maps the net output to a position; default ``tanh`` (positions in
        ``[-1, 1]``).
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
        position_fn: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        seed: int = 0,
    ):
        self.net = net
        self.layers = tuple(layers)
        self.loss = loss if loss is not None else SharpeLoss()
        self.optimizer_cls = optimizer
        self.lr = lr
        self.epochs = epochs
        self.position_fn = position_fn
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

    def fit(self, X: NDArray, y: NDArray) -> ObjectiveModel:
        """ Train the net to maximize the objective of ``positions * y``.

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

        self.net.train()  # type: ignore[union-attr]
        for _ in range(self.epochs):
            self._optim.zero_grad()  # type: ignore[union-attr]
            strat_ret = self._positions(Xt) * rt
            loss = self.loss(strat_ret)
            loss.backward()
            self._optim.step()  # type: ignore[union-attr]

        return self

    @torch.no_grad()
    def predict(self, X: NDArray) -> NDArray:
        """ Return positions in ``[-1, 1]`` for each row of ``X``. """
        self.net.eval()  # type: ignore[union-attr]
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))

        return self._positions(Xt).reshape(-1, 1).cpu().numpy()
