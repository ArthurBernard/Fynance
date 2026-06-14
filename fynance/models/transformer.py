#!/usr/bin/env python3
# coding: utf-8

""" Transformer model for financial sequences.

Defines :class:`Transformer`, a causal Transformer encoder on
:class:`~fynance.models._base.BaseNeuralNet`. It reuses the attention
building blocks from :mod:`fynance.models.attention` and applies a
**causal mask** so position ``t`` attends only to ``≤ t`` — preserving
the library's no-lookahead invariant for time-series prediction.

Main entry points
-----------------
- :class:`PositionalEncoding` — sinusoidal absolute positional encoding.
- :class:`Transformer` — stacked causal Transformer encoder blocks.

References
----------
.. [1] Vaswani, A. et al. (2017). Attention Is All You Need.

"""

from __future__ import annotations

# Built-in packages
import math

# Third-party packages
import polars as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray

# Local packages
from fynance.models._base import BaseNeuralNet
from fynance.models.attention import MultiHeadAttention

__all__ = ['PositionalEncoding', 'Transformer']


class PositionalEncoding(nn.Module):
    r""" Sinusoidal absolute positional encoding (Vaswani et al., 2017).

    Adds a fixed position-dependent signal to the input embeddings so the
    attention layers can use the order of the sequence.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int, optional
        Maximum supported sequence length. Default ``5000``.

    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ Add positional encoding to ``x`` of shape ``(B, T, d_model)``. """
        return x + self.pe[: x.size(1)]


class _TransformerBlock(nn.Module):
    """ Causal Transformer encoder block: self-attention + feed-forward. """

    def __init__(self, d_model, num_heads, dim_ff, drop):
        super().__init__()
        # MultiHeadAttention already applies the residual + layer-norm.
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=drop)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dim_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        x, _ = self.attn(x, mask=mask)

        return self.norm(x + self.drop(self.ff(x)))


class Transformer(BaseNeuralNet):
    r""" Causal Transformer encoder for sequential financial data.

    Projects the ``N`` input features to ``d_model``, adds sinusoidal
    positional encoding, applies ``num_layers`` causal self-attention
    blocks, and reads out to ``M`` outputs. A lower-triangular mask makes
    every block **strictly causal** (no lookahead): the output at ``t``
    depends only on inputs up to ``t``.

    Configure the optimizer with :meth:`BaseNeuralNet.set_optimizer`
    (e.g. with :class:`fynance.models.loss.SharpeLoss`).

    Parameters
    ----------
    X, y : array-like or int
        - If array-like, respectively the input and output data.
        - If an integer, respectively the input and output dimension.
    d_model : int, optional
        Embedding / model dimension (divisible by ``num_heads``). Default 32.
    num_heads : int, optional
        Number of attention heads. Default 4.
    num_layers : int, optional
        Number of stacked encoder blocks. Default 2.
    dim_ff : int, optional
        Hidden size of the position-wise feed-forward sublayer. Default 64.
    drop : float, optional
        Dropout probability. Default 0.

    See Also
    --------
    fynance.models.attention.MultiHeadAttention, fynance.models.tcn.TemporalConvNet

    Examples
    --------
    >>> import torch
    >>> from fynance.models.transformer import Transformer
    >>> _ = torch.manual_seed(0)
    >>> X = torch.randn(40, 3)
    >>> y = torch.randn(40, 1)
    >>> model = Transformer(X, y, d_model=16, num_heads=2, num_layers=2)
    >>> model(X).shape
    torch.Size([40, 1])

    """

    def __init__(
        self,
        X: NDArray | torch.Tensor | pl.DataFrame | int,
        y: NDArray | torch.Tensor | pl.DataFrame | int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_ff: int = 64,
        drop: float = 0.,
        x_type=None,
        y_type=None,
    ):
        """ Initialize object. """
        BaseNeuralNet.__init__(self)

        if isinstance(X, int) and isinstance(y, int):
            self.N, self.M = X, y

        else:
            self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)  # type: ignore[arg-type]

        self.input_proj = nn.Linear(self.N, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, num_heads, dim_ff, drop)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, self.M)

    def forward(self, x):
        """ Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input window, shape ``(L, N)``.

        Returns
        -------
        torch.Tensor
            Per-step output, shape ``(L, M)``.

        """
        length = x.size(0)
        z = self.input_proj(x).unsqueeze(0)  # (1, L, d_model)
        z = self.pos_encoder(z)
        # Lower-triangular causal mask: position t may attend to <= t only.
        mask = torch.tril(torch.ones(length, length, device=x.device))
        for block in self.blocks:
            z = block(z, mask)

        return self.output_proj(z.squeeze(0))  # (L, M)
