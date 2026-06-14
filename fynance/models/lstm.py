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
        self.W_f = nn.Linear(self.N + self.H, self.C)
        self.f_f = forget_activation()

        # Update gate
        self.W_i = nn.Linear(self.N + self.H, self.C)
        self.f_i = update_activation()

        # Candidate value
        self.W_c = nn.Linear(self.N + self.H, self.C)
        self.f_c = memory_activation()

        # Output gate
        self.W_o = nn.Linear(self.N + self.H, self.C)
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
    """ Long Short-Term Memory neural network.

    Full LSTM model: :class:`_LSTMCell` four-gate architecture followed
    by a forward output projection. The cell state ``C`` and hidden
    state ``H`` are threaded through the sequence, allowing the model to
    carry information across many time steps without the
    vanishing-gradient pathology that limits
    :class:`~fynance.models.rnn.RecurrentNeuralNetwork`. Use it for
    sequence modeling tasks where dependencies span dozens of steps
    (intraday return series, multi-day momentum signals, regime
    detection).

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
    forward_activation : torch.nn.Module, optional
        Activation functions, default is Softmax.
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
        forward_activation=nn.Softmax, hidden_activation=nn.Tanh,
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
        self.optimizer.zero_grad()  # type: ignore[attr-defined]
        outputs, H, C = self(X, H, C)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()  # type: ignore[attr-defined]

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss, H.detach(), C.detach()

    @torch.no_grad()
    def predict(self, X: torch.Tensor, H: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """ Predicts outputs of neural network model.

        Parameters
        ----------
        X : torch.Tensor
            Inputs to compute prediction.
        H : torch.Tensor
            States of the model.
        C : torch.Tensor
            Cell memory of the model.

        Returns
        -------
        torch.Tensor
            Outputs prediction.
        torch.Tensor
            Updated states of the model.
        torch.Tensor
            Cell memory of the model.

        """
        Y, H, C = self(X, H, C)

        return Y.detach(), H.detach(), C.detach()
