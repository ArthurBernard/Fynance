#!/usr/bin/env python3
# coding: utf-8

""" Long Short-Term Memory (LSTM) model.

Defines :class:`LSTMCell`, a composable LSTM building block, and
:class:`LongShortTermMemory`, a full LSTM model with output projection.
The internal :class:`_LSTMCell` holds the four LSTM gates (forget,
input/update, candidate, output) and is the common base for both.

The distinction mirrors PyTorch's own ``torch.nn.LSTMCell`` vs
``torch.nn.LSTM``: :class:`LSTMCell` is the raw cell (useful for
composing larger architectures such as TCN or Transformer encoders),
while :class:`LongShortTermMemory` wraps it with an output projection
and training helpers.

Main entry points
-----------------
- :class:`LSTMCell` — composable LSTM cell without output projection.
- :class:`LongShortTermMemory` — LSTM model ready for walk-forward
  training via :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`.

References
----------
.. [1] Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
       Neural Computation, 9(8), 1735–1780.

"""

from __future__ import annotations

# Third-party packages
import torch
from torch import nn

# Local packages
from fynance.models._recurrent_base import _OutputLayerMixin, _RecurrentBase

__all__ = ['LSTMCell', 'LongShortTermMemory']


class _LSTMCell(_RecurrentBase):
    """ LSTM cell: four gates without output projection.

    Implements the Long Short-Term Memory forward pass (Hochreiter &
    Schmidhuber, 1997) with forget gate ``G_f``, input gate ``G_i``,
    candidate cell ``C_tild``, and output gate ``G_o``. Returns the
    updated hidden state ``H`` and cell state ``C`` — no output layer.
    Use :class:`LongShortTermMemory` for a complete model with output
    projection and training helpers.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
    hidden_activation, memory_activation : torch.nn.Module, optional
        Activation functions for respectively hidden and memory state,
        default both are Tanh function.
    hidden_state_size, memory_state_size : int, optional
        Size of respectively hidden and memory states. Default hidden
        state is the same size as input; default memory state is the
        same size as hidden state.
    forget_activation, update_activation, output_activation :
    torch.nn.Module, optional
        Activation functions for respectively forget, update and output
        gate, default are Sigmoid function for all three.

    Attributes
    ----------
    W_f, W_i, W_o, W_c : torch.nn.Linear
        Respectively forget, update and output gate weights and weight to
        compute the candidate value for cell memory.
    f_f, f_i, f_o, f_c : torch.nn.Module
        Respectively activation function for forget, update and output gate
        and activation function to compute the candidate value for cell
        memory.

    See Also
    --------
    LongShortTermMemory,
    fynance.models.gru._GRUCell

    """

    def __init__(
        self, X, y=None, drop=None, x_type=None, y_type=None, bias=True,
        hidden_activation=nn.Tanh, hidden_state_size=None,
        memory_activation=nn.Tanh, memory_state_size=None,
        forget_activation=nn.Sigmoid, update_activation=nn.Sigmoid,
        output_activation=nn.Sigmoid,
    ):

        _RecurrentBase.__init__(
            self,
            X,
            y,
            drop=drop,
            x_type=x_type,
            y_type=y_type,
            bias=bias,
            hidden_activation=hidden_activation,
            hidden_state_size=hidden_state_size,
        )

        self.C = self.H if memory_state_size is None else memory_state_size

        # Forget gate
        self.W_f = nn.Linear(self.N + self.H, self.C, bias=bias)
        self.f_f = forget_activation()

        # Update gate
        self.W_i = nn.Linear(self.N + self.H, self.C, bias=bias)
        self.f_i = update_activation()

        # Candidate value
        self.W_c = nn.Linear(self.N + self.H, self.C, bias=bias)
        self.f_c = memory_activation()

        # Output gate
        self.W_o = nn.Linear(self.N + self.H, self.C, bias=bias)
        self.f_o = output_activation()

        # Hidden activation (applied to cell state before output gate)
        self.f_h = hidden_activation()

    def forward(self, X, H, C):
        X_H = torch.cat([X, H], dim=1)

        # Forget gate
        G_f = self.f_f(self.W_f(self.drop(X_H)))

        # Candidate value
        C_tild = self.f_c(self.W_c(self.drop(X_H)))

        # Update gate
        G_i = self.f_i(self.W_i(self.drop(X_H)))

        C = G_f * C + G_i * C_tild

        # Output gate
        G_o = self.f_o(self.W_o(self.drop(X_H)))

        H = G_o * self.f_h(C)

        return H, C


class LSTMCell(_LSTMCell):
    """ LSTM cell — public composable building block.

    Implements the four LSTM gates (forget, input, candidate, output)
    without an output projection layer. Designed to be composed inside
    larger architectures (TCN, Transformers, encoder-decoders). For a
    standalone trainable model with output projection, use
    :class:`LongShortTermMemory`.

    Parameters
    ----------
    X : int or array-like
        Input dimension (int) or input data. When passing an int, ``y``
        may be omitted.
    y : array-like or int, optional
        Output data or output dimension. Not required when using the
        cell as a building block.
    hidden_state_size : int, optional
        Size of the hidden state. Defaults to the input size.
    memory_state_size : int, optional
        Size of the cell state. Defaults to hidden state size.
    drop : float, optional
        Dropout probability applied before each gate.
    hidden_activation, memory_activation : torch.nn.Module, optional
        Activations for hidden and cell state (default: Tanh for both).
    forget_activation, update_activation, output_activation :
    torch.nn.Module, optional
        Gate activations (default: Sigmoid for all three).

    Examples
    --------
    >>> import torch
    >>> from fynance.models.lstm import LSTMCell
    >>> cell = LSTMCell(8, hidden_state_size=16)
    >>> H = torch.zeros(4, 16)
    >>> C = torch.zeros(4, 16)
    >>> X = torch.randn(4, 8)
    >>> H_new, C_new = cell(X, H, C)
    >>> H_new.shape
    torch.Size([4, 16])

    See Also
    --------
    LongShortTermMemory : full model with output projection and training.
    fynance.models.gru.GRUCell : GRU variant.

    """

    def train_on(self, *args, **kwargs):
        raise NotImplementedError(
            "LSTMCell is a composable building block with no output projection. "
            "Use LongShortTermMemory for a standalone trainable model."
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "LSTMCell is a composable building block with no output projection. "
            "Use LongShortTermMemory for a standalone trainable model."
        )


class LongShortTermMemory(_OutputLayerMixin, LSTMCell):
    """ Long Short-Term Memory cell with output projection.

    LSTM four-gate architecture (:class:`_LSTMCell`) followed by a
    forward output projection. The caller supplies the hidden state ``H``
    and cell state ``C``; :meth:`forward` returns updated ``(Y, H, C)``.
    Like the other gated cells in this package, **each of the ``T`` rows
    of ``X`` is processed independently** — the cell does *not* loop over
    a time axis or thread ``H`` / ``C`` across rows on its own, so it is a
    *stateless* gated feed-forward cell. To model temporal dependencies
    the caller must thread ``H`` and ``C`` across successive steps
    explicitly. For built-in, causal sequence modeling prefer
    :class:`~fynance.models.tcn.TemporalConvNet` or
    :class:`~fynance.models.transformer.Transformer`.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
    bias : bool, optional
        If ``True`` (default), the linear layers learn an additive bias.
    forward_activation : torch.nn.Module, optional
        Output activation, default is Identity (unconstrained regression
        output; pass ``nn.Softmax`` for a probability-simplex output).
    hidden_activation, memory_activation : torch.nn.Module, optional
        Activation functions for respectively hidden and memory state,
        default both are Tanh function.
    hidden_state_size, memory_state_size : int, optional
        Size of respectively hidden and memory states. Default hidden
        state is the same size as input; default memory state is the
        same size as hidden state.
    forget_activation, update_activation, output_activation :
    torch.nn.Module, optional
        Activation functions for respectively forget, update and output
        gate, default are Sigmoid function for all three.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_f, W_i, W_o, W_c, W_y : torch.nn.Linear
        Respectively forget, update and output gate weights, weight to
        compute the candidate value for cell memory and forward weight.
    f_f, f_i, f_o, f_c, f_y : torch.nn.Module
        Respectively activation function for forget, update and output gate,
        activation function to compute the candidate value for cell memory
        and forward activation function.

    See Also
    --------
    fynance.models.rnn.RecurrentNeuralNetwork,
    fynance.models.gru.GatedRecurrentUnit

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        forward_activation=nn.Identity, hidden_activation=nn.Tanh,
        hidden_state_size=None, memory_activation=nn.Tanh,
        memory_state_size=None, forget_activation=nn.Sigmoid,
        update_activation=nn.Sigmoid, output_activation=nn.Sigmoid,
    ):

        LSTMCell.__init__(
            self,
            X,
            y,
            drop=drop,
            x_type=x_type,
            y_type=y_type,
            bias=bias,
            hidden_activation=hidden_activation,
            hidden_state_size=hidden_state_size,
            memory_activation=memory_activation,
            memory_state_size=memory_state_size,
            forget_activation=forget_activation,
            update_activation=update_activation,
            output_activation=output_activation,
        )

        _OutputLayerMixin.__init__(self, forward_activation=forward_activation)

    def forward(self, X, H, C):
        """ Forward method.

        Parameters
        ----------
        X, H, C : torch.Tensor
            Respectively input data, hidden state and memory state.

        Returns
        -------
        torch.Tensor
            Output data.
        torch.Tensor
            Hidden state.
        torch.Tensor
            Memory state.

        """
        H, C = super().forward(X, H, C)
        Y = self.f_y(self.W_y(self.drop(H)))

        return Y, H, C

    def fit(self, X, y, epochs: int = 1, x_type=None, y_type=None):
        """ Fit the model on ``(X, y)`` for ``epochs`` full-batch steps.

        Conforms to the :class:`~fynance.core.protocols.SignalModel`
        contract. The hidden state ``H`` and cell state ``C`` are both
        zero-initialized once and threaded across epochs (detached
        between steps). An optimizer must have been registered with
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
        LongShortTermMemory
            ``self``, to allow chaining.

        """
        self.set_data(X, y, x_type=x_type, y_type=y_type)
        H = self._init_state(self.X)
        C = self._init_cell_state(self.X)

        for _ in range(epochs):
            _, H, C = self.train_on(self.X, self.y, H, C)

        return self

    def _init_cell_state(self, X: torch.Tensor) -> torch.Tensor:
        """ Build a zero cell state matching ``X`` (rows, dtype, device). """
        try:
            param = next(self.parameters())
            device, dtype = param.device, param.dtype

        except StopIteration:
            device, dtype = X.device, X.dtype

        return torch.zeros(X.shape[0], self.C, dtype=dtype, device=device)

    @torch.enable_grad()
    def train_on(self, X: torch.Tensor, y: torch.Tensor, H: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """ Trains the neural network model.

        Parameters
        ----------
        X, y, H, C : torch.Tensor
            Respectively inputs, outputs, states and cell memory to train
            model.

        Returns
        -------
        torch.nn.modules.loss
            Loss outputs.
        torch.Tensor
            Updated states of the model.
        torch.Tensor
            Cell memory of the model.

        """
        self.train()
        self.optimizer.zero_grad()  # type: ignore[attr-defined]
        outputs, H, C = self(X, H, C)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()  # type: ignore[attr-defined]

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss, H.detach(), C.detach()

    @torch.no_grad()
    def predict(self, X, H: torch.Tensor | None = None, C: torch.Tensor | None = None):  # type: ignore[override]
        """ Predicts outputs of neural network model.

        Two calling conventions are supported:

        - ``predict(X)`` — conforms to the
          :class:`~fynance.core.protocols.SignalModel` contract: ``X``
          may be array-like (coerced to a tensor), the hidden state and
          cell state are zero-initialized, and **only** the prediction
          tensor ``Y`` is returned.
        - ``predict(X, H, C)`` — explicit-state form: the updated states
          are threaded back, returning the ``(Y, H, C)`` tuple. ``C`` is
          zero-initialized when omitted.

        In both cases ``X`` (and any supplied state) is moved to the
        model's device.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Inputs to compute prediction.
        H : torch.Tensor, optional
            States of the model. If ``None`` (default), a zero state is
            used and only the prediction is returned.
        C : torch.Tensor, optional
            Cell memory of the model. Zero-initialized when ``None``.

        Returns
        -------
        torch.Tensor
            Outputs prediction (when ``H`` is ``None``).
        tuple of torch.Tensor
            ``(Y, H, C)`` outputs prediction and updated states (when
            ``H`` is provided).

        """
        return_state = H is not None

        if not isinstance(X, torch.Tensor):
            X = self._set_data(X)

        try:
            device = next(self.parameters()).device
            X = X.to(device)
            if H is not None:
                H = H.to(device)
            if C is not None:
                C = C.to(device)

        except StopIteration:
            pass

        if H is None:
            H = self._init_state(X)

        if C is None:
            C = self._init_cell_state(X)

        was_training = self.training
        self.eval()
        try:
            Y, H, C = self(X, H, C)

        finally:
            self.train(was_training)

        if return_state:

            return Y.detach(), H.detach(), C.detach()

        return Y.detach()
