#!/usr/bin/env python3
# coding: utf-8

""" Internal building blocks for the gated cell models.

Defines two private classes shared by the gated-cell variants
(:mod:`fynance.models.rnn`, :mod:`fynance.models.gru`,
:mod:`fynance.models.lstm`):

- :class:`_RecurrentBase` — shared backbone that sets up the weight
  matrix ``W_h``, the hidden activation ``f_h``, and dropout. Its
  ``forward(X, H)`` implements one Elman-style step (concatenate input
  with the supplied hidden state, apply linear + activation). **Each row
  of the input is processed independently**: the backbone does not loop
  over a time axis or thread state across rows — it is a *stateless*
  gated feed-forward cell. :class:`~fynance.models.gru._GRUCell` and
  :class:`~fynance.models.lstm._LSTMCell` override ``forward`` to add
  their gating logic on top of this backbone.

- :class:`_OutputLayerMixin` — mixin that adds the output projection
  layer ``W_y`` / ``f_y`` and provides shared ``train_on(X, y, H)``
  and ``predict(X, H)`` for models whose ``forward`` returns ``(Y, H)``.
  :class:`~fynance.models.lstm.LongShortTermMemory` overrides both
  because its signature carries an extra cell-state argument ``C``.

These classes are **not part of the public API** and must not be
instantiated directly. Use :class:`~fynance.models.rnn.RecurrentNeuralNetwork`,
:class:`~fynance.models.gru.GatedRecurrentUnit`, or
:class:`~fynance.models.lstm.LongShortTermMemory`.

"""

from __future__ import annotations

# Third-party packages
import torch
from torch import nn

# Local packages
from fynance.models._base import BaseNeuralNet


class _RecurrentBase(BaseNeuralNet):
    """ Shared backbone for the gated cell models.

    Sets up the weight matrix ``W_h``, the hidden-state activation
    ``f_h``, and the dropout layer. Its ``forward(X, H)`` implements one
    Elman-style step: concatenate input with the supplied hidden state,
    apply ``W_h`` + ``f_h``. Each row of ``X`` is processed
    independently against the matching row of ``H``; no state is threaded
    across rows, so this is a *stateless* gated feed-forward cell rather
    than a sequence model.

    :class:`~fynance.models.gru._GRUCell` and
    :class:`~fynance.models.lstm._LSTMCell` subclass this and override
    ``forward`` to replace the Elman step with their respective gating
    logic. :class:`~fynance.models.rnn.RecurrentNeuralNetwork` inherits
    the Elman ``forward`` unchanged and only adds the output projection
    (via :class:`_OutputLayerMixin`).

    Parameters
    ----------
    X : array-like or int
        Input data (array-like) or input dimension (int).
    y : array-like or int, optional
        Output data (array-like) or output dimension (int). Pass ``None``
        when instantiating a cell-only building block that has no output
        projection (e.g. :class:`~fynance.models.gru.GRUCell`).
    drop : float, optional
        Probability of an element to be zeroed.
    bias : bool, optional
        If ``True`` (default), the linear layers learn an additive bias.
    hidden_activation : torch.nn.Module, optional
        Activation functions, default is Tanh function.
    hidden_state_size : int, optional
        Size of hidden states, default is the same size than input.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_h : torch.nn.Linear
        Recurrent weights.
    f_h : torch.nn.Module
        Hidden activation function.

    See Also
    --------
    fynance.models._base.BaseNeuralNet,
    fynance.models.rnn.RecurrentNeuralNetwork,
    fynance.models.gru.GatedRecurrentUnit,
    fynance.models.lstm.LongShortTermMemory

    """

    def __init__(
        self,
        X: torch.Tensor | int,
        y: torch.Tensor | int | None = None,
        drop: float | None = None,
        x_type=None,
        y_type=None,
        bias: bool = True,
        hidden_activation: type[nn.Module] = nn.Tanh,
        hidden_state_size: int | None = None,
    ):
        BaseNeuralNet.__init__(self)

        if y is None:
            self.N = X if isinstance(X, int) else X.shape[1]
            self.M = None
        elif isinstance(X, int) and isinstance(y, int):
            self.N, self.M = X, y
        else:
            self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)  # type: ignore[arg-type]

        self.bias = bias
        self.H = self.N if hidden_state_size is None else hidden_state_size

        self.W_h = nn.Linear(self.N + self.H, self.H, bias=bias)

        self.f_h = hidden_activation()

        self.drop = self._set_dropout(drop)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        C = torch.cat([X, H], dim=1)

        return self.f_h(self.W_h(self.drop(C)))

    def _set_dropout(self, drop):
        # Set dropout parameters
        if drop is not None:

            return torch.nn.Dropout(p=drop)

        else:

            return lambda x: x


class _OutputLayerMixin:
    """ Mixin that adds an output projection layer to a recurrent model.

    Provides the output weight ``W_y`` and activation ``f_y``, and
    implements ``train_on(X, y, H)`` / ``predict(X, H)`` for models
    whose ``forward`` returns ``(Y, H)``. Classes that carry an extra
    cell state (e.g. :class:`~fynance.models.lstm.LongShortTermMemory`)
    must override both methods.

    For drop-in :class:`~fynance.core.protocols.SignalModel` use the
    mixin also provides :meth:`fit` and a single-argument
    :meth:`predict`. Because each row is processed independently
    (stateless gated cell), the hidden state has no carry-over meaning
    across a fit / predict call and the natural default is to
    zero-initialize it: :meth:`fit` zero-initializes ``H`` and threads
    it across epochs, and :meth:`predict` called as ``predict(X)``
    zero-initializes ``H`` and returns only the prediction. The
    explicit-state forms ``train_on(X, y, H)`` and ``predict(X, H)``
    remain available for callers that thread the state themselves.

    This mixin is designed for multiple inheritance alongside
    :class:`_RecurrentBase` or one of its subclasses. The MRO must
    place :class:`_OutputLayerMixin` **before** the recurrent base so
    that ``train_on`` / ``predict`` defined here take precedence over
    those inherited from :class:`~fynance.models._base.BaseNeuralNet`.

    Attributes
    ----------
    W_y : torch.nn.Linear
        Output projection weights.
    f_y : torch.nn.Module
        Output activation function.

    See Also
    --------
    fynance.models.rnn.RecurrentNeuralNetwork,
    fynance.models.gru.GatedRecurrentUnit,
    fynance.models.lstm.LongShortTermMemory

    """

    def __init__(self, forward_activation=nn.Identity):
        self.W_y = nn.Linear(self.H, self.M, bias=getattr(self, 'bias', True))
        self.f_y = nn.Softmax(dim=-1) if forward_activation is nn.Softmax else forward_activation()

    def _init_state(self, X: torch.Tensor) -> torch.Tensor:
        """ Build a zero hidden state matching ``X`` (rows, dtype, device). """
        try:
            param = next(self.parameters())  # type: ignore[attr-defined]
            device, dtype = param.device, param.dtype

        except StopIteration:
            device, dtype = X.device, X.dtype

        return torch.zeros(X.shape[0], self.H, dtype=dtype, device=device)  # type: ignore[attr-defined]

    def fit(self, X, y, epochs: int = 1, x_type=None, y_type=None):
        """ Fit the model on ``(X, y)`` for ``epochs`` full-batch steps.

        Conforms to the :class:`~fynance.core.protocols.SignalModel`
        contract. The hidden state is zero-initialized once and threaded
        across epochs (detached between steps). An optimizer must have
        been registered with
        :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`.

        Parameters
        ----------
        X, y : array-like
            Input and output data (numpy / torch / polars), shapes
            ``(T, N)`` and ``(T, M)``.
        epochs : int
            Number of full-batch training steps.
        x_type, y_type : torch.dtype, optional
            Target dtypes forwarded to
            :meth:`~fynance.models._base.BaseNeuralNet.set_data`.

        Returns
        -------
        _OutputLayerMixin
            ``self``, to allow chaining.

        """
        self.set_data(X, y, x_type=x_type, y_type=y_type)  # type: ignore[attr-defined]
        H = self._init_state(self.X)  # type: ignore[attr-defined]

        for _ in range(epochs):
            _, H = self.train_on(self.X, self.y, H)  # type: ignore[attr-defined]

        return self

    @torch.enable_grad()
    def train_on(self, X: torch.Tensor, y: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Trains the neural network model.

        Parameters
        ----------
        X, y, H : torch.Tensor
            Respectively inputs, outputs and states to train model.

        Returns
        -------
        torch.nn.modules.loss
            Loss outputs.
        torch.Tensor
            Updated states of the model.

        """
        self.train()  # type: ignore[attr-defined]
        self.optimizer.zero_grad()  # type: ignore[attr-defined]
        outputs, H = self(X, H)  # type: ignore[operator]
        loss = self.criterion(outputs, y)  # type: ignore[attr-defined]
        loss.backward()
        self.optimizer.step()  # type: ignore[attr-defined]

        if self.lr_scheduler:  # type: ignore[attr-defined]
            self.lr_scheduler.step()  # type: ignore[attr-defined]

        return loss, H.detach()

    @torch.no_grad()
    def predict(self, X, H: torch.Tensor | None = None):
        """ Predicts outputs of neural network model.

        Two calling conventions are supported:

        - ``predict(X)`` — conforms to the
          :class:`~fynance.core.protocols.SignalModel` contract: ``X``
          may be array-like (coerced to a tensor), the hidden state is
          zero-initialized, and **only** the prediction tensor ``Y`` is
          returned.
        - ``predict(X, H)`` — explicit-state form: ``X`` and ``H`` are
          tensors and the updated state is threaded back, returning the
          ``(Y, H)`` tuple.

        In both cases ``X`` (and ``H``) are moved to the model's device.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Inputs to compute prediction.
        H : torch.Tensor, optional
            States of the model. If ``None`` (default), a zero state is
            used and only the prediction is returned.

        Returns
        -------
        torch.Tensor
           Outputs prediction (when ``H`` is ``None``).
        tuple of torch.Tensor
           ``(Y, H)`` outputs prediction and updated state (when ``H``
           is provided).

        """
        return_state = H is not None

        if not isinstance(X, torch.Tensor):
            X = self._set_data(X)  # type: ignore[attr-defined]

        try:
            device = next(self.parameters()).device  # type: ignore[attr-defined]
            X = X.to(device)
            if return_state:
                H = H.to(device)  # type: ignore[union-attr]

        except StopIteration:
            pass

        if H is None:
            H = self._init_state(X)

        was_training = self.training  # type: ignore[attr-defined]
        self.eval()  # type: ignore[attr-defined]
        try:
            Y, H = self(X, H)  # type: ignore[operator]

        finally:
            self.train(was_training)  # type: ignore[attr-defined]

        if return_state:

            return Y.detach(), H.detach()

        return Y.detach()
