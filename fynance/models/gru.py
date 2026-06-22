#!/usr/bin/env python3
# coding: utf-8

""" Gated Recurrent Unit (GRU) model.

Defines :class:`GRUCell`, a composable GRU building block, and
:class:`GatedRecurrentUnit`, a full GRU model with output projection.
The internal :class:`_GRUCell` holds the GRU gating logic (reset +
update gates) and is the common base for both.

The distinction mirrors PyTorch's own ``torch.nn.GRUCell`` vs
``torch.nn.GRU``: :class:`GRUCell` is the raw cell (useful for
composing larger architectures such as TCN or Transformer encoders),
while :class:`GatedRecurrentUnit` wraps it with an output projection
and training helpers.

Main entry points
-----------------
- :class:`GRUCell` — composable GRU cell without output projection.
- :class:`GatedRecurrentUnit` — GRU model ready for walk-forward
  training via :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`.

References
----------
.. [1] Cho, K. et al. (2014). Learning Phrase Representations using
       RNN Encoder-Decoder for Statistical Machine Translation.

"""

from __future__ import annotations

# Third-party packages
import torch
from torch import nn

# Local packages
from fynance.models._recurrent_base import _OutputLayerMixin, _RecurrentBase

__all__ = ['GRUCell', 'GatedRecurrentUnit']


class _GRUCell(_RecurrentBase):
    """ GRU cell: reset and update gates without output projection.

    Implements the Gated Recurrent Unit forward pass (Cho et al., 2014)
    with reset gate ``G_r`` and update gate ``G_u``. Returns the updated
    hidden state ``H`` — no output layer. Use :class:`GatedRecurrentUnit`
    for a complete model with output projection and training helpers.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
    hidden_activation : torch.nn.Module, optional
        Activation function for the candidate hidden state, default is Tanh.
    hidden_state_size : int, optional
        Size of hidden states, default is the same size than input.
    reset_activation, update_activation : torch.nn.Module, optional
        Activation functions for reset and update gate, default are both
        Sigmoid function.

    Attributes
    ----------
    W_h, W_u, W_r : torch.nn.Linear
        Respectively recurrent (candidate), update and reset gate weights.
    f_h, f_u, f_r : torch.nn.Module
        Respectively candidate, update and reset gate activation functions.

    See Also
    --------
    GatedRecurrentUnit,
    fynance.models.lstm._LSTMCell

    """

    def __init__(
        self, X, y=None, drop=None, x_type=None, y_type=None, bias=True,
        hidden_activation=nn.Tanh, hidden_state_size=None,
        reset_activation=nn.Sigmoid, update_activation=nn.Sigmoid,
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

        self.W_u = nn.Linear(self.N + self.H, self.H, bias=bias)
        self.W_r = nn.Linear(self.N + self.H, self.H, bias=bias)

        self.f_u = update_activation()
        self.f_r = reset_activation()

    def forward(self, X, H):
        C = torch.cat([X, H], dim=1)

        # Update gate
        G_u = self.f_u(self.W_u(self.drop(C)))

        # Reset gate
        G_r = self.f_r(self.W_r(self.drop(C)))

        C_tild = torch.cat([X, G_r * H], dim=1)
        H_tild = self.f_h(self.W_h(self.drop(C_tild)))

        return G_u * H_tild + (1 - G_u) * H


class GRUCell(_GRUCell):
    """ GRU cell — public composable building block.

    Implements the GRU gating logic (reset + update gates) without an
    output projection layer. Designed to be composed inside larger
    architectures (TCN, Transformers, encoder-decoders). For a
    standalone trainable model with output projection, use
    :class:`GatedRecurrentUnit`.

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
    drop : float, optional
        Dropout probability applied before each gate.
    hidden_activation : torch.nn.Module, optional
        Activation for the candidate hidden state (default: Tanh).
    reset_activation, update_activation : torch.nn.Module, optional
        Gate activations (default: Sigmoid for both).

    Examples
    --------
    >>> import torch
    >>> from fynance.models.gru import GRUCell
    >>> cell = GRUCell(8, hidden_state_size=16)
    >>> H = torch.zeros(4, 16)
    >>> X = torch.randn(4, 8)
    >>> H_new = cell(X, H)
    >>> H_new.shape
    torch.Size([4, 16])

    See Also
    --------
    GatedRecurrentUnit : full model with output projection and training.
    fynance.models.lstm.LSTMCell : LSTM variant.

    """

    def train_on(self, *args, **kwargs):
        raise NotImplementedError(
            "GRUCell is a composable building block with no output projection. "
            "Use GatedRecurrentUnit for a standalone trainable model."
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "GRUCell is a composable building block with no output projection. "
            "Use GatedRecurrentUnit for a standalone trainable model."
        )


class GatedRecurrentUnit(_OutputLayerMixin, GRUCell):  # type: ignore[misc]
    """ Gated Recurrent Unit cell with output projection.

    GRU gating logic (:class:`_GRUCell`) followed by a forward output
    projection. Like :class:`~fynance.models.rnn.RecurrentNeuralNetwork`,
    this is a *stateless* gated feed-forward cell: each of the ``T`` rows
    of ``X`` is processed independently against the supplied hidden state
    ``H``, and no state is threaded across rows. The gating mitigates the
    vanishing-gradient pathology *within* a step but does not, on its
    own, model temporal dependencies — the caller must thread ``H``
    explicitly. For built-in sequence modeling use
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
    forward_activation, hidden_activation : torch.nn.Module, optional
        Activation functions, default is respectively Identity and Tanh
        function. The output activation defaults to Identity so the cell
        produces unconstrained regression outputs (pass ``nn.Softmax`` for
        a probability-simplex output).
    hidden_state_size : int, optional
        Size of hidden states, default is the same size than input.
    reset_activation, update_activation : torch.nn.Module, optional
        Activation functions for reset and update gate, default are both
        Sigmoid function.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_h, W_r, W_u, W_y : torch.nn.Linear
        Respectively recurrent (candidate), reset, update and forward weights.
    f_h, f_r, f_u, f_y : torch.nn.Module
        Respectively candidate, reset, update and forward activation functions.

    See Also
    --------
    fynance.models.rnn.RecurrentNeuralNetwork,
    fynance.models.lstm.LongShortTermMemory

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        forward_activation=nn.Identity, hidden_activation=nn.Tanh,
        hidden_state_size=None, reset_activation=nn.Sigmoid,
        update_activation=nn.Sigmoid,
    ):

        GRUCell.__init__(
            self,
            X,
            y,
            drop=drop,
            x_type=x_type,
            y_type=y_type,
            bias=bias,
            hidden_activation=hidden_activation,
            hidden_state_size=hidden_state_size,
            reset_activation=reset_activation,
            update_activation=update_activation,
        )

        _OutputLayerMixin.__init__(self, forward_activation=forward_activation)

    def forward(self, X, H):
        """ Forward method.

        Parameters
        ----------
        X, H : torch.Tensor
            Respectively input data and hidden state.

        Returns
        -------
        torch.Tensor
            Output data.
        torch.Tensor
            Hidden state.

        """
        H = super().forward(X, H)
        Y = self.f_y(self.W_y(self.drop(H)))

        return Y, H
