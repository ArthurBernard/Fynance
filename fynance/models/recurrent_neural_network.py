#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-06-16 08:27:56
# @Last modified by: ArthurBernard
# @Last modified time: 2023-07-31 17:30:04

""" Recurrent Neural Network models. """

# Built-in packages

# Third party packages
import torch
from torch import nn

# Local packages
from fynance.models.neural_network import BaseNeuralNet

__all__ = ['RecurrentNeuralNetwork', 'GatedRecurrentUnit',
           'LongShortTermMemory']


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

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_h : torch.nn.Linear
        Recurrent wheights.
    f_h : torch.nn.Module
        Activation functions.

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

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_h, W_y : torch.nn.Linear
        Respectively recurrent and forward wheights.
    f_y, f_h : torch.nn.Module
        Respectively forward and hidden activation functions.

    See Also
    --------
    GatedRecurrentUnit, LongShortTermMemory

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
        H = super(RecurrentNeuralNetwork, self).forward(X, H)
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

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_c, W_r, W_u : torch.nn.Linear
        Respectively recurrent (hidden), reset and update wheights.
    f_h, f_r, f_u : torch.nn.Module
        Respectively hidden (recurrent), reset, and update activation
        functions.

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


class GatedRecurrentUnit(_ForwardLayer, _GatedRecurrentUnit):
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

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_c, W_r, W_u, W_y : torch.nn.Linear
        Respectively recurrent (hidden), reset, update and forward wheights.
    f_h, f_r, f_u, f_y : torch.nn.Module
        Respectively hidden (recurrent), reset, update and forward activation
        functions.

    See Also
    --------
    RecurrentNeuralNetwork, LongShortTermMemory

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
    forget_activation, updated_activation, output_activation : torch.nn.Module,
    optional
        Activation functions for respectively forget, update and output gate,
        default are Sigmoid function for the three.

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_f, W_i, W_o, W_c : torch.nn.Linear
        Respectively forget, update and output gate weights and weight to
        compute the candidate value for cell memory.
    f_f, f_i, f_o, f_c : torch.nn.Module
        Respectively activation function for forget, update and output gate and
        activation function to compute the candidate value for cell memory.

    See Also
    --------
    BaseNeuralNet, MultiLayerPerceptron

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        hidden_activation=nn.Tanh, hidden_state_size=None,
        memory_activation=nn.Tanh, memory_state_size=None,
        forget_activation=nn.Sigmoid, update_activation=nn.Sigmoid,
        output_activation=nn.Sigmoid,
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

        # Set forget gate
        self.W_f = nn.Linear(self.N + self.H, self.C)
        self.f_f = forget_activation()

        # Set update gate
        self.W_i = nn.Linear(self.N + self.H, self.C)
        self.f_i = update_activation()

        # Set weight and activation for candidate value
        self.W_c = nn.Linear(self.N + self.H, self.C)
        self.f_c = memory_activation()

        # Set output gate
        self.W_o = nn.Linear(self.N + self.H, self.C)
        self.f_o = output_activation()

        # Set hidden activation
        self.f_h = hidden_activation()

    def forward(self, X, H, C):
        # C = torch.cat([X, H], dim=1)
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


class LongShortTermMemory(_ForwardLayer, _LongShortTermMemory):
    """ Long short term memory neural network.

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
        Activation functions for respectively hidden and memory state, default
        both are Tanh function.
    hidden_state_size, memory_state_size : int, optional
        Size of respectively hidden and memory states, default hidden state is
        the same size than input and default memory state is the same size than
        hidden state.
    forget_activation, updated_activation, output_activation : torch.nn.Module,
    optional
        Activation functions for respectively forget, update and output gate,
        default are Sigmoid function for the three.

    Methods
    -------
    __call__
    set_optimizer
    train_on
    predict
    set_data

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
        activation function to compute the candidate value for cell memory and
        forward activation function.

    See Also
    --------
    RecurrentNeuralNetwork, GatedRecurrentUnit

    """

    def __init__(
        self, X, y, drop=None, x_type=None, y_type=None, bias=True,
        forward_activation=nn.Softmax, hidden_activation=nn.Tanh,
        hidden_state_size=None, memory_activation=nn.Tanh,
        memory_state_size=None, forget_activation=nn.Sigmoid,
        update_activation=nn.Sigmoid, output_activation=nn.Sigmoid,
    ):

        _LongShortTermMemory.__init__(
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

        _ForwardLayer.__init__(self, forward_activation=forward_activation)

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
        H, C = super(LongShortTermMemory, self).forward(X, H, C)
        Y = self.f_y(self.W_y(self.drop(H)))

        return Y, H, C

    @torch.enable_grad()
    def train_on(self, X, y, H, C):
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
        self.optimizer.zero_grad()
        outputs, H, C = self(X, H, C)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss, H.detach(), C.detach()

    @torch.no_grad()
    def predict(self, X, H, C):
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


if __name__ == "__main__":
    pass
