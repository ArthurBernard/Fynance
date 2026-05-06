#!/usr/bin/env python3
# coding: utf-8

""" Attention mechanisms for sequential financial data.

PyTorch implementations of the building blocks introduced by
Vaswani et al. (2017): scaled dot-product attention and multi-head
attention. Used to build Transformer-style architectures for return
series, order books and other sequential financial signals where
long-range dependencies matter.

Main entry points
-----------------
- :class:`ScaledDotProductAttention` — single-head scaled attention.
- :class:`MultiHeadAttention` — parallel attention heads with
  learnable projections.

References
----------
.. [1] Vaswani, A. et al. (2017). Attention Is All You Need.

"""

import math

from torch import nn

__all__ = ['ScaledDotProductAttention', 'MultiHeadAttention']


class ScaledDotProductAttention(nn.Module):
    r""" Scaled Dot-Product Attention.

    Computes :math:`\text{Attention}(Q, K, V) =
    \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V`.

    Parameters
    ----------
    dropout : float, optional
        Dropout probability applied to the attention weights, default 0.

    References
    ----------
    Vaswani et al., "Attention is All You Need", arXiv 2017.

    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """ Compute attention.

        Parameters
        ----------
        Q : torch.Tensor
            Queries, shape ``(B, ..., T, d_k)``.
        K : torch.Tensor
            Keys, shape ``(B, ..., S, d_k)``.
        V : torch.Tensor
            Values, shape ``(B, ..., S, d_v)``.
        mask : torch.Tensor, optional
            Boolean mask of shape ``(B, ..., T, S)``.  Positions where
            ``mask == 0`` are set to ``-inf`` before softmax.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, ..., T, d_v)``.
        torch.Tensor
            Attention weights of shape ``(B, ..., T, S)``.

        """
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(self.softmax(scores))
        return attn @ V, attn


class MultiHeadAttention(nn.Module):
    r""" Multi-Head Self-Attention.

    Building block of the Transformer architecture (Vaswani et al.,
    2017). Each attention head learns to attend to a different subspace
    of the input — useful when several types of dependency coexist in
    a sequence, e.g. short-term and long-term momentum. Outputs of the
    heads are concatenated and projected back through ``w_o``;
    residual connection plus layer norm stabilize training.

    For finance-specific use, this layer is typically stacked with a
    feed-forward sublayer to form a Transformer encoder block applied
    to a return / order-book sequence.

    Splits the input into ``num_heads`` heads, applies
    :class:`ScaledDotProductAttention` in parallel, then re-projects.  A
    residual connection and layer norm are applied.

    Parameters
    ----------
    d_model : int
        Model dimension (must be divisible by ``num_heads``).
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout on attention weights and output projection, default 0.

    Examples
    --------
    >>> import torch
    >>> mha = MultiHeadAttention(64, 4)
    >>> x = torch.randn(2, 10, 64)
    >>> out, attn = mha(x)
    >>> out.shape
    torch.Size([2, 10, 64])
    >>> attn.shape
    torch.Size([2, 4, 10, 10])

    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f'd_model ({d_model}) must be divisible by num_heads ({num_heads})'
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        """ Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, d_model)``.
        mask : torch.Tensor, optional
            Attention mask of shape ``(B, 1, T, T)`` or ``(B, T, T)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, d_model)``.
        torch.Tensor
            Averaged attention weights of shape ``(B, num_heads, T, T)``.

        """
        B, T, _ = x.shape

        Q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask=mask)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(out)

        return self.norm(x + self.dropout(out)), attn
