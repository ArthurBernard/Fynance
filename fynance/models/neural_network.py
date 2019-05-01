#!/usr/bin/env python3
# coding: utf-8

# Built-in packages

# External packages
try:
    import torch
    import torch.nn as nn
except ImportError:
    print('You must install torch package')
    import sys
    sys.exit(0)

# Internal packages

__all__ = ['NeuralNetwork']


class NeuralNetwork(nn.Module):
    """ Base object for neural network model with pytorch as nn.Module object
    with some higher level methods.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self):
        """ Initialize """
        nn.Module.__init__(self)

    def set_optimizer(self, criterion, optimizer, **kwargs):
        """ Setting optimizer object for all parameters (weights and bias)
        defined in `forward` method.

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
        """ Training neural network.

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
        """ Outputs prediction of model.

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
