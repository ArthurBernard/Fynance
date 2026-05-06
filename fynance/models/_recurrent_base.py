#!/usr/bin/env python3
# coding: utf-8

""" Internal building blocks for recurrent neural network models.

Defines two private classes used by all RNN variants:

- :class:`_RecurrentBase` — shared backbone that sets up the recurrent
  weight matrix ``W_h``, the hidden activation ``f_h``, and dropout.
  Its ``forward(X, H)`` implements the vanilla Elman RNN step
  (concatenate input with hidden state, apply linear + activation).
  :class:`~fynance.models.gru._GRUCell` and
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
    """ Shared recurrent backbone for all RNN-flavored models.

    Sets up the recurrent weight matrix ``W_h``, the hidden-state
    activation ``f_h``, and the dropout layer. Its ``forward(X, H)``
    implements the vanilla Elman RNN step: concatenate input with hidden
    state, apply ``W_h`` + ``f_h``.

    :class:`~fynance.models.gru._GRUCell` and
    :class:`~fynance.models.lstm._LSTMCell` subclass this and override
    ``forward`` to replace the Elman step with their respective gating
    logic. :class:`~fynance.models.rnn.RecurrentNeuralNetwork` inherits
    the Elman ``forward`` unchanged and only adds the output projection
    (via :class:`_OutputLayerMixin`).

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
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
        y: torch.Tensor | int,
        drop: float | None = None,
        x_type=None,
        y_type=None,
        bias: bool = True,
        hidden_activation: type[nn.Module] = nn.Tanh,
        hidden_state_size: int | None = None,
    ):
        BaseNeuralNet.__init__(self)

        if isinstance(X, int) and isinstance(y, int):
            self.N, self.M = X, y

        else:
            self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)

        self.H = self.N if hidden_state_size is None else hidden_state_size

        self.W_h = nn.Linear(self.N + self.H, self.H)

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

    def __init__(self, forward_activation=nn.Softmax):
        self.W_y = nn.Linear(self.H, self.M)
        self.f_y = nn.Softmax(dim=-1) if forward_activation is nn.Softmax else forward_activation()

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
        self.optimizer.zero_grad()
        outputs, H = self(X, H)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss, H.detach()

    @torch.no_grad()
    def predict(self, X: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Predicts outputs of neural network model.

        Parameters
        ----------
        X : torch.Tensor
            Inputs to compute prediction.
        H : torch.Tensor
            States of the model.

        Returns
        -------
        torch.Tensor
           Outputs prediction.
        torch.Tensor
           Updated states of the model.

        """
        Y, H = self(X, H)

        return Y.detach(), H.detach()
