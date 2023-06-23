#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-06-16 08:27:56
# @Last modified by: ArthurBernard
# @Last modified time: 2023-06-23 20:02:01

""" Recurrent Neural Network models. """

# Built-in packages

# Third party packages
import torch
from torch import nn

# Local packages
from fynance.models.neural_network import BaseNeuralNet

__all__ = []


class _RecurrentNeuralNetwork(BaseNeuralNet):
    """ Neural network with recurrent architecture.

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
    W_h
        Recurrent wheights.
    f_h : torch.nn.Module, optional
        Activation functions.

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    See Also
    --------
    BaseNeuralNet, MultiLayerPerceptron

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        hidden_activation=nn.Tanh, hidden_state_size=None,
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

    def forward(self, X, H):
        C = torch.cat([X, H], dim=1)

        return self.f_h(self.W_h(self.drop(C)))

    def _set_dropout(self, drop):
        # Set dropout parameters
        if drop is not None:

            return torch.nn.Dropout(p=drop)

        else:

            return lambda x: x

    @torch.enable_grad()
    def train_on(self, X, y, H):
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
        outputs = self(X, H)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss, outputs.detach()

    @torch.no_grad()
    def predict(self, X, H):
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
        return self(X, H).detach()


class _ForwardLayer:
    def __init__(self, forward_activation=nn.Softmax):
        self.W_y = nn.Linear(self.H, self.M)
        self.f_y = forward_activation()

    @torch.enable_grad()
    def train_on(self, X, y, H):
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
    def predict(self, X, H):
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


class RecurrentNeuralNetwork(_ForwardLayer, _RecurrentNeuralNetwork):
    """ Neural network with recurrent architecture.

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
    W_h, W_y
        Respectively recurrent and forward wheights.
    f_y, f_h : torch.nn.Module, optional
        Respectively forward and hidden activation functions.

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    See Also
    --------
    BaseNeuralNet, MultiLayerPerceptron

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        forward_activation=nn.Softmax, hidden_activation=nn.Tanh,
        hidden_state_size=None,
    ):
        _RecurrentNeuralNetwork.__init__(
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

        _ForwardLayer.__init__(self, forward_activation=forward_activation)

    def forward(self, X, H):
        H = super(_RecurrentNeuralNetwork, self).forward(X, H)
        Y = self.f_y(self.W_y(self.drop(H)))

        return Y, H


class _GatedRecurrentUnit(_RecurrentNeuralNetwork):
    """ Gated Recurrent Unit neural network.

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
    reset_activation, updated_activation : torch.nn.Module, optional
        Activation functions for reset and update gate, default are both
        Sigmoid function.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_c, W_r, W_u
        Respectively recurrent (hidden), reset and update wheights.
    f_h, f_r, f_u : torch.nn.Module, optional
        Respectively hidden (recurrent), reset, and update activation
        functions.

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    See Also
    --------
    BaseNeuralNet, MultiLayerPerceptron

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        hidden_activation=nn.Tanh, hidden_state_size=None,
        reset_activation=nn.Sigmoid, update_activation=nn.Sigmoid,
    ):

        _RecurrentNeuralNetwork.__init__(
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

        self.W_u = nn.Linear(self.N + self.H, self.H)
        self.W_r = nn.Linear(self.N + self.H, self.H)

        self.f_u = update_activation()
        self.f_r = reset_activation()

    def forward(self, X, H):
        C = torch.cat([X, H], dim=1)

        # Update gate
        G_u = self.f_u(self.W_u(self.drop(C)))

        # Reset gate
        G_r = self.f_r(self.W_r(self.drop(C)))

        C_tild = torch.cat([X, G_r * H])
        H_tild = self.f_h(self.W_h(self.drop(C_tild)))

        return G_u * H_tild + (1 - G_u) * H


class GatedRecurrentUnit(BaseNeuralNet):
    """ Gated Recurrent Unit neural network.

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
    reset_activation, updated_activation : torch.nn.Module, optional
        Activation functions for reset and update gate, default are both
        Sigmoid function.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_c, W_r, W_u, W_y
        Respectively recurrent (hidden), reset, update and forward wheights.
    f_h, f_r, f_u, f_y : torch.nn.Module, optional
        Respectively hidden (recurrent), reset, update and forward activation
        functions.

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    See Also
    --------
    BaseNeuralNet, MultiLayerPerceptron

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        forward_activation=nn.Softmax, hidden_activation=nn.Tanh,
        hidden_state_size=None, reset_activation=nn.Sigmoid,
        update_activation=nn.Sigmoid,
    ):

        _GatedRecurrentUnit.__init__(
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

        _ForwardLayer.__init__(self, forward_activation=forward_activation)

    def forward(self, X, H):
        H = super(_GatedRecurrentUnit, self).forward(X, H)
        Y = self.f_y(self.W_y(self.drop(H)))

        return Y, H


class _LongShortTermMemory(_RecurrentNeuralNetwork):
    """ Long short term memory neural network.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
    hidden_activation, memory_activation : torch.nn.Module, optional
        Activation functions for respectively hidden and memory state, default
        both are Tanh function.
    hidden_state_size, memory_state_size : int, optional
        Size of respectively hidden and memory states, default hidden state is
        the same size than input and default memory state is the same size than
        hidden state.
    reset_activation, updated_activation : torch.nn.Module, optional
        Activation functions for reset and update gate, default are both
        Sigmoid function.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_c, W_r, W_u
        Respectively recurrent (hidden), reset and update wheights.
    f_h, f_r, f_u : torch.nn.Module, optional
        Respectively hidden (recurrent), reset, and update activation
        functions.

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    See Also
    --------
    BaseNeuralNet, MultiLayerPerceptron

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        hidden_activation=nn.Tanh, hidden_state_size=None,
        memory_activation=nn.Tanh, memory_state_size=None,
        reset_activation=nn.Sigmoid, update_activation=nn.Sigmoid,
    ):

        _RecurrentNeuralNetwork.__init__(
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
        self.f_c = memory_activation()

        # self.W_u = nn.Linear(self.N + self.H, self.H)
        # self.W_r = nn.Linear(self.N + self.H, self.H)

        # self.f_u = update_activation()
        # self.f_r = reset_activation()

    def forward(self, X, H):
        # C = torch.cat([X, H], dim=1)

        # # Update gate
        # G_u = self.f_u(self.W_u(self.drop(C)))

        # # Reset gate
        # G_r = self.f_r(self.W_r(self.drop(C)))

        # C_tild = torch.cat([X, G_r * H])
        # H_tild = self.f_h(self.W_h(self.drop(C_tild)))

        # return G_u * H_tild + (1 - G_u) * H
        pass


if __name__ == "__main__":
    pass
