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
``X``; ``predict(X)`` returns positions. With the default ``position_fn``
(``tanh``) these are bounded in ``[-1, 1]``; a custom ``position_fn`` may be
unbounded.

**Single-asset and panel.** With ``N`` assets the net outputs a **position
book** ``(T, N)`` — one column per asset — and is trained on the objective of
the **aggregated book return** ``(positions * returns).sum(axis=1)``. The
single-asset path (``N == 1``) is unchanged: ``X`` of shape ``(T, F)`` with a
1-D ``y`` of shape ``(T,)`` still trains exactly as before and ``predict``
returns ``(T, 1)``. For a panel, pass either a 3-D ``X`` of shape
``(T, N, M)`` (``M`` features per asset, flattened internally to ``(T, N*M)``
for the default dense net) or a pre-flattened 2-D ``X`` of shape ``(T, N*M)``
together with a 2-D ``y`` of shape ``(T, N)`` (the per-asset returns)::

    # N = 3 assets, M = 4 features each
    model = ObjectiveModel(n_assets=3, loss=SharpeLoss(), epochs=80)
    model.fit(X, y)            # X (T, 3, 4) or (T, 12); y (T, 3)
    book = model.predict(X)    # positions, shape (T, 3)

"""

# Built-in
from __future__ import annotations

import copy
import os
from typing import Any, Callable, Sequence

# Third-party
import numpy as np
import torch
from numpy.typing import NDArray

# Local
from fynance.models.loss import SharpeLoss

__all__ = ['ObjectiveModel', 'pretrain_pooled']

# Training hyper-parameters that :func:`pretrain_pooled` and
# :meth:`ObjectiveModel.finetune` accept as per-call ``**fit_kw`` overrides; each
# maps to an identically named instance attribute, set for the run only and then
# restored.
_TRAIN_OVERRIDES = frozenset(
    {"epochs", "lr", "batch_size", "shuffle", "cost", "seed"}
)


def _default_net(
    n_features: int, layers: tuple[int, ...], n_assets: int = 1,
) -> torch.nn.Module:
    """ A plain feed-forward net with ReLU hidden layers and a linear head.

    The head is ``Linear(dim, n_assets)`` so the net outputs one position per
    asset; ``n_assets == 1`` reproduces the original single-asset head exactly.
    """
    mods: list[torch.nn.Module] = []
    dim = n_features
    for h in layers:
        mods += [torch.nn.Linear(dim, h), torch.nn.ReLU()]
        dim = h
    mods += [torch.nn.Linear(dim, n_assets)]  # linear position-book head

    return torch.nn.Sequential(*mods)


class ObjectiveModel:
    """ Train a net to maximize a differentiable financial objective.

    Parameters
    ----------
    net : torch.nn.Module, optional
        Architecture mapping a feature matrix ``(T, F)`` to a position book
        ``(T, N)`` (``N`` = number of assets; ``(T, 1)`` for the single-asset
        case). Defaults to an MLP built lazily on the first :meth:`fit` (so it
        learns ``F`` and ``N``). Pass any ``nn.Module`` (e.g. a TCN/LSTM) to use
        a custom architecture; a custom net receives ``X`` as the 2-D matrix
        ``(T, F)`` (a 3-D panel ``(T, N, M)`` is flattened to ``(T, N*M)`` first).
    n_assets : int, optional
        Number of assets ``N`` in the position book. ``None`` (default) infers
        it at :meth:`fit`: from the 2nd dimension of a 2-D ``y`` ``(T, N)``, or
        from the 2nd dimension of a 3-D ``X`` ``(T, N, M)``, falling back to
        ``1`` for a 1-D ``y`` (the single-asset case).
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
        n_assets: int | None = None,
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
        self.n_assets = n_assets
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
        # Batch plan of the last training pass: a list of ``(segment, a, b)``
        # slices, each fully inside one segment. Exposed for introspection /
        # tests (a mini-batch never spans a segment/asset join by construction).
        self._batch_plan: list[tuple[int, int, int]] = []

    def _ensure_net(self, n_features: int) -> None:
        if self.net is None:
            torch.manual_seed(self.seed)
            self.net = _default_net(n_features, self.layers, self.n_assets or 1)
        if self._optim is None:
            self._optim = self.optimizer_cls(
                self.net.parameters(), lr=self.lr,  # type: ignore[call-arg]
            )

    def _positions(self, X: torch.Tensor) -> torch.Tensor:
        """ Position book ``(T, N)`` for the feature matrix ``X`` ``(T, F)``.

        ``N`` is :attr:`n_assets` (``1`` for the single-asset case). The net
        output is reshaped to ``(T, N)`` before ``position_fn`` so a single-asset
        run yields the same numbers as the original flat ``(T,)`` path.
        """
        n = self.n_assets or 1
        out = self.net(X).reshape(-1, n)  # type: ignore[misc]

        return self.position_fn(out)

    def _strat_return(self, pos: torch.Tensor, ret: torch.Tensor,
                      prev: torch.Tensor | None) -> torch.Tensor:
        """ Net-of-cost **book** return for a chunk, as a 1-D series ``(T,)``.

        ``pos`` and ``ret`` are ``(T, N)`` (a single-asset run uses ``N == 1``).
        The per-asset net-of-cost return is ``pos*ret - cost*|Δpos|`` and the
        book return aggregates across assets (sum over the asset axis), so the
        1-D series handed to the (1-D) loss is the aggregated book return.

        ``prev`` is the (detached) last position book of the previous contiguous
        chunk so the turnover at the chunk boundary is charged correctly per
        asset; ``None`` (or a shuffled chunk) charges entry from flat on the
        first bar.
        """
        strat = pos * ret
        if self.cost:
            first = pos[:1].abs() if prev is None else (pos[:1] - prev).abs()
            turnover = torch.cat([first, torch.abs(pos[1:] - pos[:-1])])
            strat = strat - self.cost * turnover

        # Aggregate the per-asset net-of-cost returns into the book return.
        return strat.sum(dim=1)

    def fit(self, X: NDArray, y: NDArray) -> ObjectiveModel:
        """ Train the net to maximize the objective of the net-of-cost return.

        Parameters
        ----------
        X : array-like, shape (T, F) or (T, N, M)
            Feature matrix. A 3-D panel ``(T, N, M)`` (``N`` assets, ``M``
            features each) is flattened to ``(T, N*M)`` for the default dense
            net.
        y : array-like, shape (T,) or (T, N)
            Realized per-bar returns aligned with ``X`` (not a supervised
            label). A 2-D ``y`` carries the per-asset returns of the position
            book; a 1-D ``y`` is the single-asset case (``N == 1``).

        Returns
        -------
        ObjectiveModel
            ``self``.

        """
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)
        n = self._resolve_n_assets(Xa, ya)
        Xt, rt = self._to_segment(Xa, ya, n)
        self._ensure_net(Xt.shape[1])
        self._run([(Xt, rt)], self._optim, self.epochs)  # type: ignore[arg-type]

        return self

    def _resolve_n_assets(self, Xa: NDArray, ya: NDArray) -> int:
        """ Resolve the asset count ``N`` (constructor value wins, else infer).

        Inference (only when :attr:`n_assets` is ``None``) reads ``N`` from the
        2nd dimension of a 2-D ``y`` ``(T, N)``, else from the 2nd dimension of a
        3-D ``X`` ``(T, N, M)``, else falls back to ``1`` (single asset). The
        resolved value is cached on :attr:`n_assets`.

        Parameters
        ----------
        Xa, ya : numpy.ndarray
            The raw (pre-flatten) feature array and return array.

        Returns
        -------
        int
            The resolved number of assets ``N``.

        """
        n = self.n_assets
        if n is None:
            if ya.ndim == 2:
                n = ya.shape[1]
            elif Xa.ndim == 3:
                n = Xa.shape[1]
            else:
                n = 1
            self.n_assets = n

        return n

    def _to_segment(
        self, X: NDArray, y: NDArray, n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Coerce one ``(X, y)`` series into aligned ``(Xt, rt)`` tensors.

        A 3-D panel ``X`` ``(T, N, M)`` is flattened to ``(T, N*M)`` for the
        dense net; the returns are reshaped to ``(T, n)``.

        Parameters
        ----------
        X : array-like, shape (T, F) or (T, N, M)
            Feature matrix for a single contiguous series.
        y : array-like, shape (T,) or (T, N)
            Realized per-bar returns aligned with ``X``.
        n : int
            Resolved number of assets ``N``.

        Returns
        -------
        tuple of torch.Tensor
            The 2-D feature tensor ``(T, F)`` and the return tensor ``(T, n)``.

        """
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)
        if Xa.ndim == 3:  # (T, N, M) panel -> (T, N*M) for the dense net
            Xa = Xa.reshape(Xa.shape[0], -1)

        return torch.as_tensor(Xa), torch.as_tensor(ya.reshape(ya.shape[0], n))

    def _segment_batches(self, T: int) -> list[tuple[int, int]]:
        """ Contiguous ``(start, stop)`` mini-batch slices for a length-``T`` series.

        Reproduces the original chunking: ``batch_size`` (or the whole series
        when ``None``) contiguous bars per batch, order preserved so the turnover
        penalty stays meaningful. Every slice lies inside the single series.
        """
        bs = self.batch_size or T
        n_chunks = (T + bs - 1) // bs

        return [(ci * bs, min((ci + 1) * bs, T)) for ci in range(n_chunks)]

    def _run(
        self,
        segments: list[tuple[torch.Tensor, torch.Tensor]],
        optim: torch.optim.Optimizer,
        epochs: int,
    ) -> ObjectiveModel:
        """ Train over one or more contiguous segments with segment-safe batching.

        A mini-batch is a contiguous slice **inside a single segment**, so no
        batch ever spans a segment (asset) join: for a single segment this is
        exactly the original :meth:`fit` loop; for several segments (see
        :func:`pretrain_pooled`) the chunks of every segment are pooled and — when
        :attr:`shuffle` — globally shuffled, but each chunk stays within its own
        segment. The turnover carry (:attr:`cost`) is reset at every segment
        boundary (entry-from-flat) and only threaded within a segment when chunks
        run in time order (no shuffle).

        Parameters
        ----------
        segments : list of tuple of torch.Tensor
            The ``(Xt, rt)`` tensor pairs, one per contiguous series.
        optim : torch.optim.Optimizer
            Optimizer stepped on each batch (``self._optim`` for
            :meth:`fit`/:func:`pretrain_pooled`; a head-only optimizer for
            :meth:`finetune`).
        epochs : int
            Number of passes over the pooled chunks.

        Returns
        -------
        ObjectiveModel
            ``self``.

        """
        chunks: list[tuple[int, int, int]] = []
        for si, (Xt, _) in enumerate(segments):
            chunks += [(si, a, b) for a, b in self._segment_batches(Xt.shape[0])]
        self._batch_plan = list(chunks)

        total = len(chunks)
        gen = torch.Generator().manual_seed(self.seed)

        self.net.train()  # type: ignore[union-attr]
        for _ in range(epochs):
            order: list[int] = list(range(total))
            if self.shuffle and total > 1:
                order = torch.randperm(total, generator=gen).tolist()

            prev: torch.Tensor | None = None
            cur_seg: int | None = None
            for k in order:
                si, a, b = chunks[k]
                if si != cur_seg:  # entering a new segment -> entry from flat
                    prev, cur_seg = None, si
                Xt, rt = segments[si]
                optim.zero_grad()
                pos = self._positions(Xt[a:b])
                # Carry the previous chunk's last position only when chunks run in
                # time order (no shuffle); a shuffled chunk charges entry-from-flat.
                strat_ret = self._strat_return(pos, rt[a:b],
                                               None if self.shuffle else prev)
                loss = self.loss(strat_ret)
                loss.backward()
                optim.step()
                prev = pos[-1].detach()

        return self

    @torch.no_grad()
    def predict(self, X: NDArray) -> NDArray:
        """ Return the position book for ``X``, shape ``(T, N)``.

        Accepts a 2-D ``X`` ``(T, F)`` or a 3-D panel ``(T, N, M)`` (flattened
        to ``(T, N*M)`` for the default net). The output is a position book with
        one column per asset (``(T, 1)`` in the single-asset case). With the
        default ``position_fn`` (``tanh``) positions are bounded in ``[-1, 1]``;
        a custom ``position_fn`` may produce unbounded values.
        """
        self.net.eval()  # type: ignore[union-attr]
        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim == 3:  # (T, N, M) panel -> (T, N*M) for the dense net
            Xa = Xa.reshape(Xa.shape[0], -1)
        Xt = torch.as_tensor(Xa)

        return self._positions(Xt).cpu().numpy()

    def _head_module(self) -> torch.nn.Module:
        """ The **head**: the last parameterized leaf module of :attr:`net`.

        The *trunk* is everything else. Leaf modules (those with no sub-modules)
        that own parameters are collected in registration order and the last one
        is taken as the head. For the default MLP this is the final ``Linear``
        position-book head, so :meth:`finetune` with ``freeze_trunk=True`` trains
        only that layer and freezes every hidden ``Linear`` (and their
        activations, which carry no parameters) before it. The rule is
        architecture-agnostic: any ``nn.Module`` whose forward path ends in a
        parameterized leaf gets a well-defined head.

        Returns
        -------
        torch.nn.Module
            The head module.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet (no net to inspect).
        ValueError
            If :attr:`net` has no parameterized leaf module.

        """
        if self.net is None:
            raise RuntimeError("no net to inspect; the model has not been fitted.")

        leaves = [
            m for m in self.net.modules()
            if not any(True for _ in m.children())
            and any(True for _ in m.parameters(recurse=False))
        ]
        if not leaves:
            raise ValueError("net has no parameterized leaf module to use as head.")

        return leaves[-1]

    def _push_overrides(
        self, kw: dict[str, Any], allowed: frozenset[str],
    ) -> dict[str, Any]:
        """ Temporarily set whitelisted training attributes, returning the old.

        Parameters
        ----------
        kw : dict
            The per-call overrides (a subset of ``allowed``).
        allowed : frozenset of str
            The attribute names that may be overridden.

        Returns
        -------
        dict
            The previous values, to hand back to :meth:`_pop_overrides`.

        Raises
        ------
        TypeError
            If ``kw`` carries a key outside ``allowed``.

        """
        unknown = set(kw) - allowed
        if unknown:
            raise TypeError(
                f"unexpected training override(s): {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}."
            )
        old = {k: getattr(self, k) for k in kw}
        for k, v in kw.items():
            setattr(self, k, v)

        return old

    def _pop_overrides(self, old: dict[str, Any]) -> None:
        """ Restore the attribute values captured by :meth:`_push_overrides`. """
        for k, v in old.items():
            setattr(self, k, v)

    def finetune(
        self,
        X: NDArray,
        y: NDArray,
        *,
        freeze_trunk: bool = True,
        lr: float | None = None,
        epochs: int | None = None,
        **fit_kw: Any,
    ) -> ObjectiveModel:
        """ Continue training from the **current** weights (no reinitialization).

        Unlike :meth:`fit` (which warm-starts but reuses the model's persistent
        optimizer and full parameter set), :meth:`finetune` builds a **fresh**
        optimizer over the currently trainable parameters, so a different ``lr``
        or a frozen trunk take effect immediately. It is the adaptation step
        after :func:`pretrain_pooled`: pretrain one net on a pool of assets, then
        :meth:`finetune` a per-asset copy (see :meth:`clone`) on that asset's own
        data.

        Parameters
        ----------
        X, y : array-like
            One contiguous ``(X, y)`` series, same shapes as :meth:`fit`.
        freeze_trunk : bool, optional
            If ``True`` (default), freeze every parameter except the head — the
            last parameterized leaf module (see :meth:`_head_module`); for the
            default MLP only the final ``Linear`` head is trained. If ``False``,
            all parameters (trunk included) keep training.
        lr : float, optional
            Learning rate for the finetuning optimizer. ``None`` (default) reuses
            the model's :attr:`lr`.
        epochs : int, optional
            Passes over the data. ``None`` (default) reuses :attr:`epochs`.
        **fit_kw
            Per-run overrides of the training hyper-parameters ``batch_size``,
            ``shuffle``, ``cost`` and ``seed`` (restored afterwards).

        Returns
        -------
        ObjectiveModel
            ``self``.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted (no weights to continue
            from).

        Examples
        --------
        >>> import numpy as np
        >>> from fynance.models import ObjectiveModel
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((16, 2)).astype("float32")
        >>> y = (X[:, 0] * 0.01).astype("float32")
        >>> m = ObjectiveModel(layers=(4,), epochs=3, seed=0).fit(X, y)
        >>> _ = m.finetune(X, y, epochs=2)      # continue from current weights
        >>> m.predict(X).shape
        (16, 1)

        """
        if self.net is None:
            raise RuntimeError(
                "finetune() requires a fitted model; call fit() first."
            )

        old = self._push_overrides(fit_kw, _TRAIN_OVERRIDES - {"lr", "epochs"})
        try:
            Xa = np.asarray(X, dtype=np.float32)
            ya = np.asarray(y, dtype=np.float32)
            n = self._resolve_n_assets(Xa, ya)
            Xt, rt = self._to_segment(Xa, ya, n)

            head_ids = {id(p) for p in self._head_module().parameters()}
            frozen: list[torch.nn.Parameter] = []
            if freeze_trunk:
                for p in self.net.parameters():
                    if id(p) not in head_ids and p.requires_grad:
                        p.requires_grad_(False)
                        frozen.append(p)

            trainable = [p for p in self.net.parameters() if p.requires_grad]
            optim = self.optimizer_cls(
                trainable, lr=self.lr if lr is None else lr,  # type: ignore[call-arg]
            )
            try:
                self._run([(Xt, rt)], optim,
                          self.epochs if epochs is None else epochs)
            finally:
                for p in frozen:  # leave the model fully trainable afterwards
                    p.requires_grad_(True)
        finally:
            self._pop_overrides(old)

        return self

    def save(self, path: str | os.PathLike) -> None:
        """ Serialize the weights and the full init config to ``path``.

        Writes (via :func:`torch.save`) the net ``state_dict`` together with
        everything needed to reconstruct the model: the net module, the loss, the
        optimizer class, the position function and the scalar hyper-parameters.
        Reload with :meth:`load`; the pair round-trips to bit-identical
        predictions and the reloaded model stays trainable.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file.

        Examples
        --------
        >>> import os, tempfile
        >>> import numpy as np
        >>> from fynance.models import ObjectiveModel
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((16, 2)).astype("float32")
        >>> y = (X[:, 0] * 0.01).astype("float32")
        >>> m = ObjectiveModel(layers=(4,), epochs=3, seed=0).fit(X, y)
        >>> path = os.path.join(tempfile.mkdtemp(), "objective.pt")
        >>> m.save(path)
        >>> restored = ObjectiveModel.load(path)
        >>> bool(np.allclose(m.predict(X), restored.predict(X), atol=1e-7))
        True

        """
        payload = {
            "state_dict": None if self.net is None else self.net.state_dict(),
            "net": self.net,
            "loss": self.loss,
            "optimizer_cls": self.optimizer_cls,
            "position_fn": self.position_fn,
            "config": {
                "n_assets": self.n_assets,
                "layers": self.layers,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "shuffle": self.shuffle,
                "cost": self.cost,
                "seed": self.seed,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> ObjectiveModel:
        """ Reconstruct a model saved by :meth:`save`.

        Rebuilds the model from the persisted init config (net, loss, optimizer
        class, position function and scalar hyper-parameters) and reloads the
        weights, yielding predictions identical to the saved model. A fresh
        optimizer is created lazily on the next :meth:`fit` / :meth:`finetune`,
        so the reloaded model is immediately trainable again.

        Parameters
        ----------
        path : str or os.PathLike
            File written by :meth:`save`.

        Returns
        -------
        ObjectiveModel
            The reconstructed model.

        See Also
        --------
        save : the round-trip example lives there.

        """
        # weights_only=False: the payload holds Python objects (net module, loss,
        # optimizer class, position function), not just tensors.
        payload = torch.load(path, weights_only=False)
        model = cls(
            net=payload["net"],
            loss=payload["loss"],
            optimizer=payload["optimizer_cls"],
            position_fn=payload["position_fn"],
            **payload["config"],
        )
        state = payload["state_dict"]
        if state is not None and model.net is not None:
            model.net.load_state_dict(state)

        return model

    def clone(self) -> ObjectiveModel:
        """ A fresh model with the **same** (deep-copied) weights, trained apart.

        The net is deep-copied so the clone starts from identical weights yet
        shares no parameter storage with the original; its optimizer is built
        lazily on first training, so fitting the clone leaves the original's
        predictions bit-identical. This is the per-asset branch of the
        pretrain/finetune workflow: :func:`pretrain_pooled` a shared net, then
        ``clone().finetune(...)`` once per asset.

        Returns
        -------
        ObjectiveModel
            The independent copy.

        Examples
        --------
        >>> import numpy as np
        >>> from fynance.models import ObjectiveModel
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((16, 2)).astype("float32")
        >>> y = (X[:, 0] * 0.01).astype("float32")
        >>> m = ObjectiveModel(layers=(4,), epochs=3, seed=0).fit(X, y)
        >>> c = m.clone()
        >>> bool(np.allclose(m.predict(X), c.predict(X), atol=1e-7))
        True
        >>> before = m.predict(X)
        >>> _ = c.fit(X, y)                             # train the clone ...
        >>> bool(np.array_equal(before, m.predict(X)))  # ... original untouched
        True

        """
        return ObjectiveModel(
            net=copy.deepcopy(self.net),
            n_assets=self.n_assets,
            layers=self.layers,
            loss=self.loss,
            optimizer=self.optimizer_cls,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            position_fn=self.position_fn,
            cost=self.cost,
            seed=self.seed,
        )

    def _fit_pooled(
        self, Xs: Sequence[NDArray], ys: Sequence[NDArray],
    ) -> ObjectiveModel:
        """ Coerce every ``(X_i, y_i)`` pair to a segment and train the pool.

        Each pair becomes one contiguous segment; :meth:`_run` then batches
        **within** segments only, so no mini-batch spans an asset join. All
        segments must share the feature dimension ``F`` and the asset count ``N``.

        Parameters
        ----------
        Xs, ys : sequence of array-like
            The aligned per-asset feature matrices and return series.

        Returns
        -------
        ObjectiveModel
            ``self``.

        Raises
        ------
        ValueError
            If the segments disagree on the feature dimension or asset count.

        """
        segments: list[tuple[torch.Tensor, torch.Tensor]] = []
        n: int | None = None
        n_features: int | None = None
        for X, y in zip(Xs, ys):
            Xa = np.asarray(X, dtype=np.float32)
            ya = np.asarray(y, dtype=np.float32)
            if n is None:
                n = self._resolve_n_assets(Xa, ya)
            Xt, rt = self._to_segment(Xa, ya, n)
            if n_features is None:
                n_features = Xt.shape[1]
            elif Xt.shape[1] != n_features:
                raise ValueError(
                    "all X_i must share the feature dimension; got "
                    f"{Xt.shape[1]} vs {n_features}."
                )
            if rt.shape[1] != n:
                raise ValueError(
                    f"all y_i must share the asset count N={n}; got {rt.shape[1]}."
                )
            segments.append((Xt, rt))

        self._ensure_net(n_features)  # type: ignore[arg-type]
        self._run(segments, self._optim, self.epochs)  # type: ignore[arg-type]

        return self


def pretrain_pooled(
    model: ObjectiveModel,
    Xs: Sequence[NDArray],
    ys: Sequence[NDArray],
    **fit_kw: Any,
) -> ObjectiveModel:
    """ Pretrain one :class:`ObjectiveModel` on a **pool** of aligned assets.

    The ``(X_i, y_i)`` pairs are pooled into a single training run so the net
    learns a signal **shared** across assets (transfer learning across a panel).
    The pooling is segment-safe: each asset's series is kept as one contiguous
    chunk and mini-batches are drawn **within** a chunk, so **no mini-batch ever
    crosses an asset join** — the turnover carry (``cost``) and the temporal
    order that the single-asset :meth:`~ObjectiveModel.fit` relies on stay intact
    per asset. Concretely this extends ``fit``'s contiguous chunking with segment
    boundaries: all segments' chunks are pooled and (when ``shuffle``) globally
    shuffled for SGD mixing, but every chunk stays inside its own segment.

    Typical use: pretrain a shared net here, then adapt per asset with
    :meth:`~ObjectiveModel.clone` + :meth:`~ObjectiveModel.finetune`.

    Parameters
    ----------
    model : ObjectiveModel
        The model to train **in place** (its hyper-parameters and seed are used).
    Xs : sequence of array-like
        Per-asset feature matrices, each ``(T_i, F)`` (or a ``(T_i, N, M)``
        panel); all must share the feature dimension.
    ys : sequence of array-like
        Per-asset realized returns aligned with ``Xs`` (``(T_i,)`` or
        ``(T_i, N)``); must have the same length as ``Xs``.
    **fit_kw
        Per-run overrides of the training hyper-parameters ``epochs``, ``lr``,
        ``batch_size``, ``shuffle``, ``cost`` and ``seed`` (restored afterwards).

    Returns
    -------
    ObjectiveModel
        The same ``model``, now pretrained on the pool.

    Raises
    ------
    ValueError
        If ``Xs`` and ``ys`` have different lengths, or the pool is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.models import ObjectiveModel, pretrain_pooled
    >>> rng = np.random.default_rng(0)
    >>> Xs = [rng.standard_normal((16, 2)).astype("float32") for _ in range(3)]
    >>> ys = [(X[:, 0] * 0.01).astype("float32") for X in Xs]
    >>> m = ObjectiveModel(layers=(4,), epochs=3, seed=0)
    >>> _ = pretrain_pooled(m, Xs, ys)      # one net trained on all three series
    >>> m.predict(Xs[0]).shape
    (16, 1)

    """
    if len(Xs) != len(ys):
        raise ValueError(
            f"Xs and ys must have the same length; got {len(Xs)} and {len(ys)}."
        )
    if not Xs:
        raise ValueError("need at least one (X, y) pair to pretrain.")

    old = model._push_overrides(fit_kw, _TRAIN_OVERRIDES)
    try:
        model._fit_pooled(Xs, ys)
    finally:
        model._pop_overrides(old)

    return model
