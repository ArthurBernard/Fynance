#!/usr/bin/env python3
# coding: utf-8

""" Feed-forward multi-layer perceptron model.

Defines :class:`MultiLayerPerceptron`, a configurable MLP built on top
of :class:`~fynance.models._base.BaseNeuralNet`. This is the standard
baseline for tabular and sliding-window features in finance.

For time-ordered sequence input, prefer the recurrent architectures in
:mod:`fynance.models.gru` or :mod:`fynance.models.lstm`. For walk-forward
training, wrap this class with
:class:`~fynance.models.rolling.RollMultiLayerPerceptron`.

Main entry points
-----------------
- :class:`MultiLayerPerceptron` — vanilla MLP with configurable hidden
  layers, activation and dropout.

"""

from __future__ import annotations

# Built-in packages
from typing import Any

# Third-party packages
import polars as pl
import torch
from numpy.typing import NDArray

# Local packages
from fynance.models._base import BaseNeuralNet

__all__ = ['MultiLayerPerceptron']


class MultiLayerPerceptron(BaseNeuralNet):
    r""" Neural network with MultiLayer Perceptron architecture.

    Refered as vanilla neural network model, with `n` hidden layers s.t
    n :math:`\geq` 1, with each one a specified number of neurons.

    Each hidden layer is a ``torch.nn.Linear`` followed by an optional
    dropout and the configured activation function. The MLP is the
    standard baseline for tabular and sliding-window features in
    finance — useful for non-linear regression on engineered features
    (technical indicators, volatility, sentiment scores). For
    time-ordered sequence input, prefer
    :class:`fynance.models.lstm.LongShortTermMemory`
    or attention-based architectures.

    Configure the optimizer with :meth:`BaseNeuralNet.set_optimizer`
    and wrap with :class:`fynance.models.rolling.RollMultiLayerPerceptron`
    for walk-forward training.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    layers : list of int
        List of number of neurons in each hidden layer.
    activation : torch.nn.Module
        Activation function of layers.
    drop : float, optional
        Probability of an element to be zeroed.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    n : int
        Number of hidden layers.
    layers : list of int
        List with the number of neurons for each hidden layer.
    f : torch.nn.Module
        Activation function.

    See Also
    --------
    fynance.models._base.BaseNeuralNet,
    fynance.models.rolling.RollMultiLayerPerceptron

    """

    def __init__(
        self,
        X: NDArray | torch.Tensor | pl.DataFrame | int,
        y: NDArray | torch.Tensor | pl.DataFrame | int,
        layers: list[int] = [],
        activation: type[torch.nn.Module] | None = None,
        drop: float | None = None,
        x_type=None,
        y_type=None,
        bias: bool = True,
        activation_kwargs: dict[str, Any] = {},
    ):
        """ Initialize object. """
        BaseNeuralNet.__init__(self)

        if isinstance(X, int) and isinstance(y, int):
            self.N, self.M = X, y

        else:
            self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)  # type: ignore[arg-type]

        self.n_layers = len(layers) + 1

        self.layers = self._set_layer_list(layers, bias)
        self.activation = self._set_activation(activation, **activation_kwargs)
        self.drop = self._set_dropout(drop)

    def _set_layer_list(self, layers, bias, input_dim=None, output_dim=None):
        layers_list = []
        # Set input layer
        input_size = self.N if input_dim is None else input_dim
        for output_size in layers:
            # Set hidden layers
            layers_list += [torch.nn.Linear(
                input_size,
                output_size,
                bias=bias
            )]
            input_size = output_size

        # Set output layer
        output_size = self.M if output_dim is None else output_dim
        layers_list += [torch.nn.Linear(input_size, output_size, bias=bias)]

        return torch.nn.ModuleList(layers_list)

    def _set_activation(self, activation, n_layers=None, **kwargs):
        # Set activation functions
        if isinstance(activation, list):
            n_layers = len(self.layers) if n_layers is None else n_layers

            if len(activation) != n_layers:

                raise ValueError('if you pass a list of activation functions '
                                 'this one must be of size of layers list + 1')

            return [a(**kwargs) for a in activation]

        elif activation is not None:

            return activation(**kwargs)

        else:

            return lambda x: x

    def _set_dropout(self, drop):
        # Set dropout parameters. Use a ModuleList (with nn.Identity for
        # the no-dropout slots) so the dropout layers are registered as
        # submodules and respond to ``model.eval()`` / ``model.train()``.
        if isinstance(drop, list):

            if len(drop) != self.n_layers:

                raise ValueError('if you pass a list of drop parameters '
                                 'this one must be of size of layers list + 1')

            modules = [torch.nn.Dropout(p=p) for p in drop]

        elif drop is not None:

            modules = [torch.nn.Dropout(p=drop) for _ in range(self.n_layers)]

        else:

            modules = [torch.nn.Identity() for _ in range(self.n_layers)]

        return torch.nn.ModuleList(modules)

    def forward(self, x):
        """ Forward computation. """
        for name, layer in enumerate(self.layers):
            x = self.drop[name](x)
            x = layer(x)

            if isinstance(self.activation, list):
                x = self.activation[name](x)

            else:
                x = self.activation(x)

        return x
