#!/usr/bin/env python3
# coding: utf-8

""" Temporal Convolutional Network (TCN) model.

Defines :class:`TemporalConvNet`, a causal dilated 1-D convolutional
network built on :class:`~fynance.models._base.BaseNeuralNet`. TCNs are a
strong sequence baseline: a stack of residual blocks with exponentially
growing dilation gives a large receptive field while staying **strictly
causal** (output at ``t`` depends only on inputs up to ``t`` — no
lookahead), which is exactly the invariant financial backtesting needs.

Reference: Bai, Kolter & Koltun, *An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling* (2018).

Main entry points
-----------------
- :class:`TemporalConvNet` — configurable causal dilated-convolution net.

"""

from __future__ import annotations

# Third-party packages
import polars as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray

# Local packages
from fynance.models._base import BaseNeuralNet

__all__ = ['TemporalConvNet']


class _Chomp1d(nn.Module):
    """ Remove the ``chomp`` trailing time steps left by causal padding. """

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        if self.chomp == 0:
            return x

        return x[:, :, :-self.chomp].contiguous()


class _TemporalBlock(nn.Module):
    """ Residual block: two causal dilated convolutions + skip connection. """

    def __init__(self, c_in, c_out, kernel_size, dilation, drop):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        # 1x1 conv to match channels on the residual path when they differ
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(BaseNeuralNet):
    r""" Causal dilated Temporal Convolutional Network.

    A stack of residual blocks (:class:`_TemporalBlock`) with dilation
    doubling at each level (1, 2, 4, …) followed by a linear read-out to
    the output dimension. Padding is causal (left-only, trimmed by
    :class:`_Chomp1d`), so the prediction at time ``t`` never sees
    ``t + 1`` — preserving the library's no-lookahead invariant.

    Configure the optimizer with :meth:`BaseNeuralNet.set_optimizer`
    (e.g. with :class:`fynance.models.loss.SharpeLoss`).

    Parameters
    ----------
    X, y : array-like or int
        - If array-like, respectively the input and output data.
        - If an integer, respectively the input and output dimension.
    channels : list of int, optional
        Number of channels of each residual block (the depth is
        ``len(channels)``; dilation of block ``i`` is ``2 ** i``).
        Default ``[16, 16]``.
    kernel_size : int, optional
        Convolution kernel size. Default ``2``.
    drop : float, optional
        Dropout probability inside each block. Default ``0.``.

    Attributes
    ----------
    tcn : torch.nn.Sequential
        The stack of temporal blocks.
    linear : torch.nn.Linear
        Final read-out from the last channel size to ``M`` outputs.

    See Also
    --------
    fynance.models._base.BaseNeuralNet, fynance.models.lstm.LongShortTermMemory

    Examples
    --------
    >>> import torch
    >>> from fynance.models.tcn import TemporalConvNet
    >>> _ = torch.manual_seed(0)
    >>> X = torch.randn(50, 3)
    >>> y = torch.randn(50, 1)
    >>> model = TemporalConvNet(X, y, channels=[8, 8], kernel_size=2)
    >>> out = model(X)
    >>> out.shape
    torch.Size([50, 1])

    """

    def __init__(
        self,
        X: NDArray | torch.Tensor | pl.DataFrame | int,
        y: NDArray | torch.Tensor | pl.DataFrame | int,
        channels: list[int] | None = None,
        kernel_size: int = 2,
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

        channels = [16, 16] if channels is None else channels
        blocks = []
        c_in = self.N
        for i, c_out in enumerate(channels):
            blocks.append(
                _TemporalBlock(c_in, c_out, kernel_size, dilation=2 ** i, drop=drop)
            )
            c_in = c_out

        self.tcn = nn.Sequential(*blocks)
        self.linear = nn.Linear(c_in, self.M)

    def forward(self, x):
        """ Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input window, shape ``(L, N)`` (time steps × features).

        Returns
        -------
        torch.Tensor
            Per-step output, shape ``(L, M)``.

        """
        # (L, N) -> (batch=1, channels=N, length=L) for Conv1d
        z = x.transpose(0, 1).unsqueeze(0)
        z = self.tcn(z)
        # (1, C, L) -> (L, C)
        z = z.squeeze(0).transpose(0, 1)

        return self.linear(z)
