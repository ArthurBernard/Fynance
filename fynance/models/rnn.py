#!/usr/bin/env python3
# coding: utf-8

""" Single-step Elman-style recurrent cell.

Defines :class:`RecurrentNeuralNetwork`, an Elman-style cell: it
concatenates an input row with a supplied hidden-state row, applies one
linear layer plus activation, and projects to the output. Each of the
``T`` rows of the input is processed **independently** — the class does
**not** thread a hidden state across a time axis, so it is a *stateless*
gated feed-forward cell rather than a sequence model. The caller is
responsible for any state threading (e.g. by looping over time steps and
feeding the returned ``H`` back in).

For genuine sequence modeling with temporal state, use a dedicated
sequence architecture such as :class:`~fynance.models.tcn.TemporalConvNet`
or :class:`~fynance.models.transformer.Transformer`, which enforce
causality over an explicit time axis.

Main entry points
-----------------
- :class:`RecurrentNeuralNetwork` — single-step Elman cell ready for
  training via :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`.

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


class RecurrentNeuralNetwork(_OutputLayerMixin, _RecurrentBase):  # type: ignore[misc]
    """ Single-step Elman-style gated cell.

    A single linear layer (over the input concatenated with a supplied
    hidden state) followed by an output projection. Each call to
    :meth:`forward` consumes ``X`` of shape ``(T, N)`` and ``H`` of shape
    ``(T, H)`` and returns ``(Y, H_new)`` with the same leading dimension
    ``T``; **every row is processed independently** and no state is
    threaded across rows. This is therefore a *stateless* gated
    feed-forward cell, not a sequence model — the caller must perform any
    temporal state threading explicitly.

    For sequence modeling with built-in temporal state and causality,
    prefer :class:`~fynance.models.tcn.TemporalConvNet` or
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

    Examples
    --------
    >>> import torch
    >>> from fynance.models.rnn import RecurrentNeuralNetwork
    >>> _ = torch.manual_seed(0)
    >>> model = RecurrentNeuralNetwork(4, 1, hidden_state_size=3)
    >>> X = torch.zeros(5, 4)
    >>> H = torch.zeros(5, 3)
    >>> Y, H_new = model(X, H)
    >>> Y.shape, H_new.shape
    (torch.Size([5, 1]), torch.Size([5, 3]))

    See Also
    --------
    fynance.models.gru.GatedRecurrentUnit,
    fynance.models.lstm.LongShortTermMemory

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        forward_activation=nn.Identity, hidden_activation=nn.Tanh,
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
