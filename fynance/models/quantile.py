#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Multi-quantile regression head.

:class:`QuantileModel` trains a feed-forward trunk with one output per
target quantile on :class:`~fynance.models.loss.PinballLoss`, giving a
**distributional** forecast instead of a single point estimate. It conforms
to the ``SignalModel`` protocol (``fit``/``predict``): ``predict`` returns
the median (or nearest-to-0.5) quantile column — the usual point-forecast
contract — while :meth:`~QuantileModel.predict_quantiles` exposes the full
quantile band.

Non-crossing quantiles
-----------------------
A net trained independently per quantile column has no structural guarantee
that ``q10 <= q50 <= q90`` at every row (:class:`~fynance.models.loss.PinballLoss`
is the plain per-column pinball loss, with no crossing penalty). Instead,
:class:`QuantileModel` enforces monotonicity **at predict time** by sorting
the raw net output along the quantile axis. ``taus`` are kept sorted
ascending internally (see
:func:`~fynance.models.loss.pinball._validate_taus`), so after this sort
column ``i`` still lines up with the ``i``-th requested quantile.

"""

# Built-in
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Third-party
import numpy as np
import torch
from numpy.typing import NDArray

# Local
from fynance.models.loss.pinball import PinballLoss, _validate_taus

__all__ = ['QuantileModel']


def _quantile_trunk(
    n_features: int,
    n_taus: int,
    layers: list[int],
    activation: type[torch.nn.Module] | None,
    activation_kwargs: dict[str, Any],
    drop: float | None,
    bias: bool,
) -> torch.nn.Module:
    r""" Feed-forward trunk: ``Linear -> [Dropout] -> [Activation]`` hidden
    blocks with a **linear** (unbounded) ``n_taus``-wide output head.

    Mirrors the architecture kwargs exposed by
    :class:`~fynance.models.mlp.MultiLayerPerceptron` (``layers``,
    ``activation``, ``drop``, ``bias``, ``activation_kwargs``), but — unlike
    that class, whose ``forward`` applies the activation to *every* layer
    including the output — leaves the output head linear, as
    :func:`fynance.models.regime_model._mlp` and
    :func:`fynance.models.objective._default_net` already do: a quantile
    regression target is real-valued and unbounded, and an output-side
    activation such as the default ``ReLU`` would clip it (verified to
    collapse training to a constant-zero net on a symmetric-around-0
    target).
    """
    mods: list[torch.nn.Module] = []
    dim = n_features
    for h in layers:
        mods.append(torch.nn.Linear(dim, h, bias=bias))
        if drop is not None:
            mods.append(torch.nn.Dropout(p=drop))
        if activation is not None:
            mods.append(activation(**activation_kwargs))
        dim = h
    mods.append(torch.nn.Linear(dim, n_taus, bias=bias))  # linear output head

    return torch.nn.Sequential(*mods)


class QuantileModel:
    r""" Multi-quantile regression ``SignalModel``.

    Trains a feed-forward trunk with one output column per ``tau`` on
    :class:`~fynance.models.loss.PinballLoss`. Unlike
    :class:`~fynance.models.objective.ObjectiveModel` (trained on
    ``positions * returns``), ``fit(X, y)`` reads ``y`` as an ordinary
    supervised target (e.g. the next-bar return), not a returns series to be
    combined with positions.

    Parameters
    ----------
    taus : float or sequence of float, optional
        Target quantile(s) in ``(0, 1)``. Default ``(0.1, 0.5, 0.9)``.
    layers : list of int, optional
        Hidden layer sizes of the trunk. Default ``[16, 8]`` (a small
        nonlinear trunk — matching the ``ObjectiveModel`` / ``RegimeMoE``
        convention of a usable-out-of-the-box default).
    activation : type[torch.nn.Module] or None, optional
        Activation class applied after each **hidden** layer (never after the
        linear output head — see :func:`_quantile_trunk`). Default
        :class:`torch.nn.ReLU`; pass ``None`` for a purely linear trunk.
    drop : float, optional
        Dropout probability after each hidden layer. Default ``None`` (no
        dropout).
    bias : bool, optional
        Whether the trunk's linear layers use a bias term. Default True.
    activation_kwargs : dict, optional
        Extra keyword arguments for ``activation``. Default ``{}``.
    lr : float, optional
        Learning rate for the optimizer. Default ``1e-3``.
    epochs : int, optional
        Full-batch training steps. Default ``200``.
    optimizer : type[torch.optim.Optimizer], optional
        Optimizer class. Default :class:`torch.optim.Adam`.
    seed : int, optional
        Seed for reproducible net initialization. Default ``0``.

    Attributes
    ----------
    net : torch.nn.Module or None
        The trunk, built lazily on the first :meth:`fit` call (``None``
        before that).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, size=300).astype(np.float32)
    >>> sigma = 0.1 + 0.3 * np.abs(x)
    >>> y = (x + sigma * rng.standard_normal(300)).astype(np.float32)
    >>> X = x.reshape(-1, 1)
    >>> model = QuantileModel(
    ...     taus=(0.1, 0.5, 0.9), layers=[16, 8], epochs=200, lr=1e-2, seed=0,
    ... ).fit(X, y)
    >>> point = model.predict(X)
    >>> point.shape
    (300,)
    >>> q = model.predict_quantiles(X)
    >>> q.shape
    (300, 3)
    >>> bool(np.all(np.diff(q, axis=1) >= 0))   # non-crossing, enforced at predict time
    True

    See Also
    --------
    fynance.models.loss.PinballLoss,
    fynance.models.objective.ObjectiveModel,
    fynance.models.regime_model.RegimeMoE

    """

    def __init__(
        self,
        taus: float | Sequence[float] = (0.1, 0.5, 0.9),
        layers: list[int] = [16, 8],
        activation: type[torch.nn.Module] | None = torch.nn.ReLU,
        drop: float | None = None,
        bias: bool = True,
        activation_kwargs: dict[str, Any] = {},
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        seed: int = 0,
    ):
        self.taus = _validate_taus(taus)
        self._median_idx = min(
            range(len(self.taus)), key=lambda i: abs(self.taus[i] - 0.5)
        )
        self.layers = list(layers)
        self.activation = activation
        self.drop = drop
        self.bias = bias
        self.activation_kwargs = dict(activation_kwargs)
        self.lr = lr
        self.epochs = epochs
        self.optimizer_cls = optimizer
        self.seed = seed
        self._loss = PinballLoss(taus=self.taus)
        self.net: torch.nn.Module | None = None
        self._optim: torch.optim.Optimizer | None = None

    def fit(self, X: NDArray, y: NDArray) -> QuantileModel:
        """ Train the trunk to minimize the multi-quantile pinball loss.

        Parameters
        ----------
        X : array-like, shape (T, F)
            Feature matrix.
        y : array-like, shape (T,) (or any shape reshaping to it)
            Supervised target aligned with ``X`` (e.g. the realized
            next-bar return) — not a returns series to combine with
            positions.

        Returns
        -------
        QuantileModel
            ``self``.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, or ``X`` and ``y`` have different lengths.

        """
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32).reshape(-1)

        if Xa.ndim != 2:
            raise ValueError(f"X must be 2-D (T, F), got shape {Xa.shape}")

        if ya.shape[0] != Xa.shape[0]:
            raise ValueError(
                "X and y must have the same length, got "
                f"{Xa.shape[0]} and {ya.shape[0]}"
            )

        # Seed before building the net for reproducible initialization.
        torch.manual_seed(self.seed)
        self.net = _quantile_trunk(
            Xa.shape[1], len(self.taus), self.layers, self.activation,
            self.activation_kwargs, self.drop, self.bias,
        )
        self._optim = self.optimizer_cls(
            self.net.parameters(), lr=self.lr,  # type: ignore[call-arg]
        )

        Xt = torch.as_tensor(Xa)
        yt = torch.as_tensor(ya)

        self.net.train()
        for _ in range(self.epochs):
            self._optim.zero_grad()
            pred = self.net(Xt)
            loss = self._loss(pred, yt)
            loss.backward()
            self._optim.step()

        return self

    @torch.no_grad()
    def predict_quantiles(self, X: NDArray) -> NDArray:
        """ Return the full quantile band, shape ``(T, n_taus)``.

        Columns follow ``self.taus`` (sorted ascending). Non-crossing is
        enforced here (not during training) by sorting the raw, independently
        trained columns along the quantile axis.

        Parameters
        ----------
        X : array-like, shape (T, F)
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Quantile predictions, shape ``(T, n_taus)``, non-decreasing
            along axis 1.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        """
        if self.net is None:
            raise RuntimeError("QuantileModel must be fit before predict_quantiles")

        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        was_training = self.net.training
        self.net.eval()
        try:
            raw = self.net(Xt)
        finally:
            self.net.train(was_training)
        sorted_q, _ = torch.sort(raw, dim=-1)

        return sorted_q.cpu().numpy()

    def predict(self, X: NDArray) -> NDArray:
        """ Point forecast: the tau=0.5 (or nearest) quantile column.

        Parameters
        ----------
        X : array-like, shape (T, F)
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Shape ``(T,)``, the median (or nearest-to-0.5 ``tau``) column of
            :meth:`predict_quantiles`.

        """
        return self.predict_quantiles(X)[:, self._median_idx]
