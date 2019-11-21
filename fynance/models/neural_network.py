#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-05-06 20:16:31
# @Last modified by: ArthurBernard
# @Last modified time: 2019-11-20 16:34:22

""" Basis of neural networks models. """

# Built-in packages

# External packages
import numpy as np
import pandas as pd
import torch
import torch.nn

# Local packages

__all__ = ['BaseNeuralNet', 'MultiLayerPerceptron']


class BaseNeuralNet(torch.nn.Module):
    """ Base object for neural network model with PyTorch.

    Inherits of torch.nn.Module object with some higher level methods.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    N, M : int
        Respectively input and output dimension.

    Methods
    -------
    set_optimizer
    train_on
    predict
    set_data

    See Also
    --------
    MultiLayerPerceptron, RollingBasis

    """

    def __init__(self):
        """ Initialize. """
        torch.nn.Module.__init__(self)

    def set_optimizer(self, criterion, optimizer, **kwargs):
        """ Set the optimizer object.

        Set optimizer object with specified `criterion` as loss function and
        any `kwargs` as optional parameters.

        Parameters
        ----------
        criterion : torch.nn.modules.loss
            A loss function.
        optimizer : torch.optim
            An optimizer algorithm.
        **kwargs
            Keyword arguments of `optimizer`, cf PyTorch documentation [1]_.

        Returns
        -------
        BaseNeuralNet
            Self object model.

        References
        ----------
        .. [1] https://pytorch.org/docs/stable/optim.html

        """
        self.criterion = criterion()
        self.optimizer = optimizer(self.parameters(), **kwargs)

        return self

    @torch.enable_grad()
    def train_on(self, X, y):
        """ Trains the neural network model.

        Parameters
        ----------
        X, y : torch.Tensor
            Respectively inputs and outputs to train model.

        Returns
        -------
        torch.nn.modules.loss
            Loss outputs.

        """
        self.optimizer.zero_grad()
        outputs = self(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def predict(self, X):
        """ Predicts outputs of neural network model.

        Parameters
        ----------
        X : torch.Tensor
           Inputs to compute prediction.

        Returns
        -------
        torch.Tensor
           Outputs prediction.

        """
        return self(X).detach()

    def set_data(self, X, y, x_type=None, y_type=None):
        """ Set data inputs and outputs.

        Parameters
        ----------
        X, y : array-like
            Respectively input and output data.
        x_type, y_type : torch.dtype
            Respectively input and ouput data types. Default is `None`.

        """
        if hasattr(self, 'N') and self.N != X.size(1):
            raise ValueError('X must have {} input columns'.foramt(self.N))

        if hasattr(self, 'M') and self.M != y.size(1):
            raise ValueError('y must have {} output columns'.format(self.M))

        self.X = self._set_data(X, dtype=x_type)
        self.y = self._set_data(y, dtype=y_type)
        self.T, self.N = self.X.size()
        T_veri, self.M = self.y.size()

        if self.T != T_veri:
            raise ValueError('{} time periods in X differents of {} time \
                             periods in y'.format(self.T, T_veri))

        return self

    def _set_data(self, X, dtype=None):
        """ Convert array-like data to tensor. """
        # TODO : Verify dtype of data torch tensor
        if isinstance(X, np.ndarray):

            return torch.from_numpy(X)

        elif isinstance(X, pd.DataFrame):
            # TODO : Verify memory efficiancy
            return torch.from_numpy(X.values)

        elif isinstance(X, torch.Tensor):

            return X

        else:
            raise ValueError('Unkwnown data type: {}'.format(type(X)))


class MultiLayerPerceptron(BaseNeuralNet):
    r""" Neural network with MultiLayer Perceptron architecture.

    Refered as vanilla neural network model, with `n` hidden layers s.t
    n :math:`\geq` 1, with each one a specified number of neurons.

    Parameters
    ----------
    X, y : array-like
        Respectively inputs and outputs data.
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

    Methods
    -------
    set_optimizer
    train_on
    predict
    set_data

    See Also
    --------
    BaseNeuralNet, RollMultiLayerPerceptron

    """

    def __init__(self, X, y, layers=[], activation=None, drop=None,
                 x_type=None, y_type=None, bias=True, activation_kwargs={}):
        """ Initialize object. """
        BaseNeuralNet.__init__(self)

        self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)
        self._set_layers(layers, bias)
        self._set_activation(activation, **activation_kwargs)
        self._set_dropout(drop)

    def _set_layers(self, layers, bias):
        layers_list = []
        # Set input layer
        input_size = self.N
        for output_size in layers:
            # Set hidden layers
            layers_list += [torch.nn.Linear(
                input_size,
                output_size,
                bias=bias
            )]
            input_size = output_size

        # Set output layer
        layers_list += [torch.nn.Linear(input_size, self.M, bias=bias)]
        self.layers = torch.nn.ModuleList(layers_list)

    def _set_activation(self, activation, **kwargs):
        # Set activation functions
        if isinstance(activation, list):

            if len(activation) != len(self.layers):

                raise ValueError('if you pass a list of activation functions '
                                 'this one must be of size of layers list + 1')

            self.activation = [a(**kwargs) for a in activation]

        elif activation is not None:
            self.activation = activation(**kwargs)

        else:
            self.activation = lambda x: x

    def _set_dropout(self, drop):
        # Set dropout parameters
        if isinstance(drop, list):

            if len(drop) != len(self.layers):

                raise ValueError('if you pass a list of drop parameters '
                                 'this one must be of size of layers list + 1')

            self.drop = [torch.nn.Dropout(p=p) for p in drop]

        elif drop is not None:
            self.drop = torch.nn.Dropout(p=drop)

        else:
            self.drop = lambda x: x

    def forward(self, x):
        """ Forward computation. """
        for name, layer in enumerate(self.layers):
            if isinstance(self.drop, list):
                x = self.drop[name](x)

            else:
                x = self.drop(x)

            x = layer(x)

            if isinstance(self.activation, list):
                x = self.activation[name](x)

            else:
                x = self.activation(x)

        return x


def _type_convert(dtype):
    if dtype is np.float64 or dtype is np.float or dtype is np.double:
        return torch.float64

    elif dtype is np.float32:
        return torch.float32

    elif dtype is np.float16:
        return torch.float16

    elif dtype is np.uint8:
        return torch.uint8

    elif dtype is np.int8:
        return torch.int8

    elif dtype is np.int16 or dtype is np.short:
        return torch.int16

    elif dtype is np.int32:
        return torch.int32

    elif dtype is np.int64 or dtype is np.int or dtype is np.long:
        return torch.int64

    else:
        raise ValueError('Unkwnown type: {}'.format(str(dtype)))
