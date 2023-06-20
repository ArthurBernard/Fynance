#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-06-16 08:27:56
# @Last modified by: ArthurBernard
# @Last modified time: 2023-06-20 16:34:06

""" Recurrent Neural Network models. """

# Built-in packages

# Third party packages
import torch
from torch import nn

# Local packages
from fynance.models.neural_network import BaseNeuralNet

__all__ = []


class RecurrentNeuralNetworkForwardCell(BaseNeuralNet):
    """ Neural network with recurrent architecture.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.
    forward_activation, hidden_activation : torch.nn.Module, optional
        Activation functions, default is respectively Softmax and Tanh function.
    hidden_state_size : int, optional
        Size of hidden states, default is the same size than input.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.
    W_c, W_y
        Respectively recurrent wheight and forward wheight.
    forward_activation, hidden_activation : torch.nn.Module, optional
        Activation functions.

    Methods
    -------
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
        BaseNeuralNet.__init__(self)

        if isinstance(X, int) and isinstance(y, int):
            self.N, self.M = X, y

        else:
            self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)

        self.H = self.N if hidden_state_size is None else hidden_state_size

        self.W_c = nn.Linear(self.N + self.H, self.H)
        self.W_y = nn.Linear(self.H, self.M)

        self.forward_activation = forward_activation()
        self.hidden_activation = hidden_activation()

        self.drop = self._set_dropout(drop)

    def forward(self, X, H):
        C = torch.cat([X, H], dim=1)

        H = self.hidden_activation(self.W_c(self.drop(C)))
        Y = self.forward_activation(self.W_y(self.drop(H)))

        return Y, H

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


class GatedRecurrentUnit(BaseNeuralNet):
    r""" Neural network with Gated Reccurent Unit architecture.

    Parameters
    ----------
    X, y : array-like or int
        - If it's an array-like, respectively inputs and outputs data.
        - If it's an integer, respectively dimension of inputs and outputs.
    drop : float, optional
        Probability of an element to be zeroed.

    Attributes
    ----------
    criterion : torch.nn.modules.loss
        A loss function.
    optimizer : torch.optim
        An optimizer algorithm.

    Methods
    -------
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
        activation_kwargs={}
    ):
        """ Initialize object. """
        BaseNeuralNet.__init__(self)

        if isinstance(X, int) and isinstance(y, int):
            self.N, self.M = X, y

        else:
            self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)

        self.H = self.N

        self.W_c = nn.Linear(self.N + self.H, self.H)
        self.W_u = nn.Linear(self.N + self.H, self.H)
        self.W_r = nn.Linear(self.N + self.H, self.H)

        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.drop = self._set_dropout(drop)

    def forward(self, X, H):
        G_u = self.sigm(self.W_u(self.drop(torch.cat((X, H)))))
        G_r = self.sigm(self.W_r(self.drop(torch.cat((X, H)))))

        h = self.tanh(self.W_c(self.drop(torch.cat((X, G_r * H)))))

        return G_u * h + (1 - G_u) * H


if __name__ == "__main__":
    pass
