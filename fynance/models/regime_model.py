#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Regime-conditioned architecture (mixture-of-experts).

:class:`RegimeMoE` conditions an objective-aligned network on the **causal**
market regime (:class:`~fynance.features.RegimeDetector`): the network's
prediction depends not only on the features but on which volatility regime the
market is currently in. Two routings are available — a learned **regime
embedding** concatenated to the features (``routing="soft"``, the differentiable
default) and one **expert head per regime** (``routing="hard"``).

It conforms to the ``SignalModel`` protocol (``fit``/``predict``) and reuses
:class:`~fynance.models.ObjectiveModel` for the training loop (differentiable
financial objective, mini-batching, net-of-cost penalty, seeding), so it plugs
into the harness via the precomputed ``X`` path exactly like ``ObjectiveModel``.

Causality
---------
The regime label is produced by a :class:`RegimeDetector` **fit on the training
slice only** and assigned **online** (nearest training centroid), so no future
information leaks into a label. The regime is derived from one designated column
of ``X`` (``regime_col``), which must be a **positive price / level** series (the
detector clusters its trailing volatility and mean return).

"""

# Built-in
from __future__ import annotations

from typing import Any

# Third-party
import numpy as np
import torch
from numpy.typing import NDArray

# Local
from fynance.features.regime import RegimeDetector
from fynance.models.objective import ObjectiveModel

__all__ = ['RegimeMoE']


def _mlp(n_in: int, hidden: tuple[int, ...]) -> torch.nn.Module:
    """ Feed-forward trunk with ReLU hidden layers and a linear (1-d) head. """
    mods: list[torch.nn.Module] = []
    dim = n_in
    for h in hidden:
        mods += [torch.nn.Linear(dim, h), torch.nn.ReLU()]
        dim = h
    mods += [torch.nn.Linear(dim, 1)]

    return torch.nn.Sequential(*mods)


class _RegimeMoENet(torch.nn.Module):
    """ Net taking augmented input ``[features | regime_label]`` -> ``(T, 1)``.

    The last input column carries the integer regime label; the rest are the
    features. ``soft`` routing embeds the regime and concatenates it to the
    features through a shared trunk; ``hard`` routing runs one expert MLP per
    regime, selected per row.
    """

    def __init__(
        self,
        n_features: int,
        n_regimes: int,
        emb_dim: int,
        hidden: tuple[int, ...],
        routing: str,
    ):
        super().__init__()
        self.n_features = n_features
        self.routing = routing

        if routing == 'soft':
            self.emb = torch.nn.Embedding(n_regimes, emb_dim)
            self.trunk = _mlp(n_features + emb_dim, hidden)

        elif routing == 'hard':
            self.experts = torch.nn.ModuleList(
                [_mlp(n_features, hidden) for _ in range(n_regimes)]
            )

        else:

            raise ValueError(f"unknown routing: {routing!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Split the regime label off the last column and route accordingly. """
        feats = x[:, :self.n_features]
        regime = x[:, self.n_features].long()

        if self.routing == 'soft':
            emb = self.emb(regime)

            return self.trunk(torch.cat([feats, emb], dim=1))

        out = torch.zeros(feats.shape[0], 1, dtype=feats.dtype)
        for k, expert in enumerate(self.experts):
            mask = regime == k

            if bool(mask.any()):
                out[mask] = expert(feats[mask])

        return out


class RegimeMoE:
    """ Regime-conditioned mixture-of-experts ``SignalModel``.

    Parameters
    ----------
    n_regimes : int
        Number of market regimes (clusters). Default 3.
    regime_col : int
        Index of the column in ``X`` used as the **positive price / level**
        series the :class:`RegimeDetector` clusters on. Default 0.
    regime_w : int
        Rolling window for the regime features. Default 21.
    regime_period : int
        Annualization factor for the regime volatility feature. Default 252.
    routing : {"soft", "hard"}
        ``soft`` (default) concatenates a learned regime **embedding** to the
        features through a shared trunk (differentiable end-to-end); ``hard``
        uses one **expert** MLP per regime, selected by the regime label.
    emb_dim : int
        Embedding size for ``soft`` routing (ignored for ``hard``). Default 4.
    hidden : tuple of int
        Hidden layer sizes of the trunk / experts. Default ``(16, 8)``.
    loss : BaseLoss, optional
        Differentiable financial loss (default :class:`SharpeLoss`).
    lr, epochs, batch_size, cost, seed
        Forwarded to :class:`~fynance.models.ObjectiveModel`.

    Notes
    -----
    Reproducible: the net is seeded (``torch.manual_seed``) **before** it is
    built, then handed to :class:`ObjectiveModel` (which would otherwise skip
    seeding a caller-provided net).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> level = 100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01))
    >>> sig = rng.choice([-1.0, 1.0], size=400)
    >>> X = np.column_stack([level, sig]).astype(np.float32)
    >>> y = (sig * 0.01).astype(np.float32)
    >>> model = RegimeMoE(n_regimes=2, regime_w=10, epochs=5).fit(X, y)
    >>> model.predict(X).shape
    (400, 1)

    """

    def __init__(
        self,
        n_regimes: int = 3,
        regime_col: int = 0,
        regime_w: int = 21,
        regime_period: int = 252,
        routing: str = 'soft',
        emb_dim: int = 4,
        hidden: tuple[int, ...] = (16, 8),
        loss: Any = None,
        lr: float = 1e-3,
        epochs: int = 80,
        batch_size: int | None = None,
        cost: float = 0.0,
        seed: int = 0,
    ):
        self.n_regimes = n_regimes
        self.regime_col = regime_col
        self.regime_w = regime_w
        self.regime_period = regime_period
        self.routing = routing
        self.emb_dim = emb_dim
        self.hidden = tuple(hidden)
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.cost = cost
        self.seed = seed
        self.detector: RegimeDetector | None = None
        self._obj: ObjectiveModel | None = None

    def _regime_series(self, X: NDArray) -> NDArray:
        """ Extract the level column used for regime detection. """
        return np.asarray(X, dtype=np.float64)[:, self.regime_col]

    def _augment(self, X: NDArray, labels: NDArray) -> NDArray:
        """ Append the regime label as the last column of ``X``. """
        return np.column_stack(
            [np.asarray(X, dtype=np.float32), labels.astype(np.float32)]
        )

    def fit(self, X: NDArray, y: NDArray) -> RegimeMoE:
        """ Fit the causal regime detector (on train) then train the MoE net.

        Parameters
        ----------
        X : array-like, shape (T, F)
            Feature matrix; column ``regime_col`` is the positive price/level
            series used for regime detection.
        y : array-like, shape (T,)
            Realized per-bar returns aligned with ``X``.

        Returns
        -------
        RegimeMoE
            ``self``.

        """
        X = np.asarray(X)
        n_features = X.shape[1]

        # Causal regime: fit the detector on the training slice only.
        self.detector = RegimeDetector(
            n_regimes=self.n_regimes, w=self.regime_w,
            period=self.regime_period, seed=self.seed,
        ).fit(self._regime_series(X))
        labels = self.detector.predict(self._regime_series(X))

        # Seed before building the net (ObjectiveModel skips seeding a given net).
        torch.manual_seed(self.seed)
        net = _RegimeMoENet(
            n_features, self.n_regimes, self.emb_dim, self.hidden, self.routing,
        )
        self._obj = ObjectiveModel(
            net=net, loss=self.loss, lr=self.lr, epochs=self.epochs,
            batch_size=self.batch_size, cost=self.cost, seed=self.seed,
        )
        self._obj.fit(self._augment(X, labels), y)

        return self

    def predict(self, X: NDArray) -> NDArray:
        """ Assign the regime online and return positions in ``[-1, 1]``. """
        if self.detector is None or self._obj is None:

            raise RuntimeError("RegimeMoE must be fit before predict")

        X = np.asarray(X)
        labels = self.detector.predict(self._regime_series(X))

        return self._obj.predict(self._augment(X, labels))
