#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-07-27 09:10:43
# @Last modified by: ArthurBernard
# @Last modified time: 2023-08-02 12:26:48

""" Neural network with attention model. """

# Built-in packages

# Third party packages
import torch
from torch import nn
# from torch.nn import functional as F

# Local packages
from fynance.models.neural_network import BaseNeuralNet

__all__ = []


class _BaseAttention(BaseNeuralNet):

    pass


class ScaledDotProductAttention(_BaseAttention):
    r""" Scaled Dot-Product Attention model.

    Attention model described in the paper "Attention is All You Need".

    .. math:: A(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V

    Parameters
    ----------
    n_q, n_k : int
        Respectively lenght of the queries and keys (and values).
    d_k, d_v : int
        Respectively dimension of the keys (and queries) and values.

    Methods
    -------
    __call__

    Attribute
    ---------
    w_q, w_k, w_v : torch.nn.Linear
        Respectively queries, keys and values weights.
    softmax : torch.nn.Softmax
        Softmax activation function.

    References
    ----------
    "Attention is All You Need" (Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia
    Polosukhin, arxiv, 2017).

    """

    def __init__(self, n_q, n_k, d_k, d_v):

        self.w_q = nn.Linear(n_q, d_k, bias=False)
        self.w_k = nn.Linear(n_k, d_k, bias=False)
        self.w_v = nn.Linear(n_k, d_v, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """ Forward method.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Queries, keys and values input data, respectively of shape (n_q,
            d_k), (n_k, d_k) and (n_k, d_v).

        Returns
        -------
        torch.Tensor
            Output of the attention model of shape (n_q, d_v).
        torch.Tensor
            Attention residual (?) of shape (n_q, n_k). # FIXME

        """
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        attn = self.dropout(self.softmax(q / q.size(1) ** 0.5 @ k.T))
        output = attn @ v

        return output, attn


class SelfAttention(ScaledDotProductAttention):

    def forward(self, x, mask=None):
        return super().forward(q=x, k=x, v=x, mask=mask)


if __name__ == "__main__":
    pass
