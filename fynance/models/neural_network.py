#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-05-06 20:16:31
# @Last modified by: ArthurBernard
# @Last modified time: 2019-05-24 08:58:07

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

    Methods
    -------
    set_optimizer(criterion, optimizer, **kwargs)
        Set optimizer object with specified criterion (loss function) and
        any optional parameters.
    train_on(X, y)
        Trains the neural network on `X` as inputs and `y` as ouputs.
    predict(X)
        Predicts the outputs of neural network model for `X` as inputs.

    See Also
    --------
    MultiLayerPerceptron

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
        kwargs : dict
            Keyword arguments of optimizer, cf pytorch documentation [1]_.

        Returns
        -------
        NeuralNetwork
            Self object model.

        References
        ----------
        .. [1] https://pytorch.org/docs/stable/optim.html

        """
        self.criterion = criterion()
        self.optimizer = optimizer(self.parameters(), **kwargs)

        return self

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


class MultiLayerPerceptron(BaseNeuralNet):
    r""" Neural network with MultiLayer Perceptron architecture.

    Refered as vanilla neural network model, with `n` hidden layers s.t
    n :math:`\geq` 1, with each one a specified number of neurons.

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
    set_optimizer(criterion, optimizer, **kwargs)
        Set optimizer object with specified criterion (loss function) and
        any optional parameters.
    train_on(X, y)
        Trains the neural network on `X` as inputs and `y` as ouputs.
    predict(X)
        Predicts the outputs of neural network model for `X` as inputs.
    set_data(X, y)
        Set respectively input and ouputs data tensor.

    See Also
    --------
    BaseNeuralNet

    """

    def __init__(self, X, y, layers=[], activation=None, drop=None):
        """ Initialize.

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

        """
        BaseNeuralNet.__init__(self)

        self.set_data(X, y)
        layers_list = []

        # Set input layer
        input_size = self.N
        for output_size in layers:
            # Set hidden layers
            layers_list += [torch.nn.Linear(input_size, output_size)]
            input_size = output_size
        else:
            # Set output layer
            layers_list += [torch.nn.Linear(input_size, self.M)]
            self.layers = torch.nn.ModuleList(layers_list)

        # Set activation functions
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = lambda x: x

        # Set dropout parameters
        if drop is not None:
            self.drop = torch.nn.Dropout(p=drop)
        else:
            self.drop = lambda x: x

    def forward(self, x):
        """ Forward computation. """
        x = self.drop(x)

        for name, layer in enumerate(self.layers):
            x = self.activation(layer(x))

        return x

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


def type_convert(dtype):
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
