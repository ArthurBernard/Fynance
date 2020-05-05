#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-05-06 20:16:31
# @Last modified by: ArthurBernard
# @Last modified time: 2020-05-05 20:25:59

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
    criterion : torch.nn.modules.loss.Loss
        A loss function.
    optimizer : torch.optim.Optimizer
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

    lr_scheduler = None
    optimizer = None
    seed_torch = None
    seed_numpy = None

    def __init__(self):
        """ Initialize. """
        torch.nn.Module.__init__(self)

    def set_optimizer(self, criterion, optimizer, params=None, **kwargs):
        """ Set the optimizer object.

        Set optimizer object with specified `criterion` as loss function and
        any `kwargs` as optional parameters.

        Parameters
        ----------
        criterion : Callabletorch.nn.modules.loss
            A loss function.
        optimizer : torch.optim.Optimizer
            An optimizer algorithm.
        params : object or iterable object
            Layer of parameters to optimize or dicts defining parameter groups.
            If set to None then all parameters of model will be optimized.
            Default is None.
        **kwargs
            Keyword arguments of ``optimizer``, cf PyTorch documentation [1]_.

        Returns
        -------
        BaseNeuralNet
            Self object model.

        References
        ----------
        .. [1] https://pytorch.org/docs/stable/optim.html

        """
        if params is None:
            params = self.parameters()

        elif isinstance(params, list):
            params = [{'params': p.parameters()} for p in params]

        else:
            params = params.parameters()

        self.criterion = criterion()
        self.optimizer = optimizer(params, **kwargs)

        return self

    def set_lr_scheduler(self, lr_scheduler, **kwargs):
        """ Set dynamic learning rate.

        Parameters
        ----------
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            Method from ``torch.optim.lr_scheduler`` to wrap
            ``self.optimizer``, cf module ``torch.optim.lr_scheduler`` in
            PyTorch documentation [2]_.
        **kwargs
            Key


        """
        if self.optimizer:
            self.lr_scheduler = lr_scheduler(self.optimizer, **kwargs)

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

        if self.lr_scheduler:
            self.lr_scheduler.step()

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

    def set_seed(self, seed_torch=None, seed_numpy=None):
        r""" Set seed for PyTorch and NumPy random number generator.

        Parameters
        ----------
        seed_torch, seed_numpy : bool or int, optional
            If `seed` is an int :math:`0 < seed < 2^32` set respectively
            PyTorch and NumPy seed with the number. Otherwise if is True
            then choose a random number, else doesn't set seed.

        """
        self.seed_torch = self._set_seed(seed_torch)
        self.seed_numpy = self._set_seed(seed_numpy)
        torch.manual_seed(self.seed_torch)
        np.random.seed(self.seed_numpy)

    def _set_seed(self, seed):
        if isinstance(seed, int) and 0 <= seed < 2 ** 32:

            return seed

        return np.random.randint(0, 2 ** 32)

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

        if isinstance(X, int) and isinstance(y, int):
            self.N, self.M = X, y

        else:
            self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)

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
        # Set dropout parameters
        if isinstance(drop, list):

            if len(drop) != len(self.layers):

                raise ValueError('if you pass a list of drop parameters '
                                 'this one must be of size of layers list + 1')

            return [torch.nn.Dropout(p=p) for p in drop]

        elif drop is not None:

            return torch.nn.Dropout(p=drop)

        else:

            return lambda x: x

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
