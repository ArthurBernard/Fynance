#!/usr/bin/env python3
# coding: utf-8

""" Vanilla Elman recurrent neural network model.

Defines :class:`RecurrentNeuralNetwork`, the simplest recurrent model:
a single recurrent layer (Elman RNN) followed by a forward output
projection. For longer temporal dependencies, prefer
:class:`~fynance.models.gru.GatedRecurrentUnit` (GRU) or
:class:`~fynance.models.lstm.LongShortTermMemory` (LSTM), which
mitigate the vanishing-gradient problem via explicit gating.

Main entry points
-----------------
- :class:`RecurrentNeuralNetwork` — vanilla Elman RNN ready for
  walk-forward training via :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`.

References
----------
.. [1] Elman, J.L. (1990). Finding Structure in Time. Cognitive Science.

"""

from __future__ import annotations

# Third-party packages
from torch import nn

# Local packages
from fynance.models._recurrent_base import _OutputLayerMixin, _RecurrentBase

__all__ = ['RecurrentNeuralNetwork']


class RecurrentNeuralNetwork(_OutputLayerMixin, _RecurrentBase):
    """ Neural network with vanilla Elman recurrent architecture.

    A single recurrent linear layer followed by a forward output layer.
    Each call to :meth:`forward` updates the hidden state ``H`` and
    emits a prediction ``Y``. Suitable as a baseline for short-horizon
    sequence prediction; for longer dependencies, use
    :class:`~fynance.models.gru.GatedRecurrentUnit` or
    :class:`~fynance.models.lstm.LongShortTermMemory` to mitigate
    vanishing gradients.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
    forward_activation, hidden_activation : torch.nn.Module, optional
        Activation functions, default is respectively Softmax and Tanh
        function.
    hidden_state_size : int, optional
        Size of hidden states, default is the same size than input.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_h, W_y : torch.nn.Linear
        Respectively recurrent and forward weights.
    f_y, f_h : torch.nn.Module
        Respectively forward and hidden activation functions.

    See Also
    --------
    fynance.models.gru.GatedRecurrentUnit,
    fynance.models.lstm.LongShortTermMemory

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        forward_activation=nn.Softmax, hidden_activation=nn.Tanh,
        hidden_state_size=None,
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
