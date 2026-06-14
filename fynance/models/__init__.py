#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-18 18:43:15
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-28 13:54:20

""" Some deep learning, econometric, statistic and/or financial models.

.. currentmodule:: fynance.models

.. toctree::

    models.mlp
    models.rnn
    models.gru
    models.lstm
    models.attention
    models.econometric_models
    models.rolling
    models.loss

"""

from . import (
    _base,
    _recurrent_base,
    attention,
    econometric_models,
    econometric_models_cy,
    ensemble,
    gru,
    loss,
    lstm,
    mlp,
    rnn,
    rolling,
    tcn,
    training,
    transformer,
)
from ._base import BaseNeuralNet
from .attention import MultiHeadAttention, ScaledDotProductAttention
from .econometric_models import ARMA, ARMA_GARCH, ARMAX_GARCH, MA, get_parameters
from .econometric_models_cy import (
    ARMA_cy,
    ARMA_GARCH_cy,
    ARMAX_GARCH_cy,
    MA_cy,
    get_parameters_cy,
)
from .ensemble import StackingEnsemble
from .gru import GatedRecurrentUnit, GRUCell
from .loss import (
    BaseLoss,
    CalmarLoss,
    DirectionalAccuracyLoss,
    HybridLoss,
    OmegaLoss,
    SharpeLoss,
    SortinoLoss,
)
from .lstm import LongShortTermMemory, LSTMCell
from .mlp import MultiLayerPerceptron
from .rnn import RecurrentNeuralNetwork
from .rolling import CVResult, RollMultiLayerPerceptron, _RollingBasis
from .tcn import TemporalConvNet
from .training import EarlyStopping, exp_sample_weights
from .transformer import PositionalEncoding, Transformer

# Frozen public surface for the 1.x series — names listed here are
# guaranteed to remain importable from ``fynance.models`` until the
# next major version. New names may be appended (additive change), but
# nothing in this list will be removed without a deprecation cycle.
__all__ = [
    # attention
    'MultiHeadAttention',
    'ScaledDotProductAttention',
    # econometric_models
    'ARMA',
    'ARMA_GARCH',
    'ARMAX_GARCH',
    'MA',
    'get_parameters',
    # econometric_models_cy
    'ARMA_cy',
    'ARMA_GARCH_cy',
    'ARMAX_GARCH_cy',
    'MA_cy',
    'get_parameters_cy',
    # _base / mlp
    'BaseNeuralNet',
    'StackingEnsemble',
    'MultiLayerPerceptron',
    # rnn / gru / lstm
    'GRUCell',
    'GatedRecurrentUnit',
    'LSTMCell',
    'LongShortTermMemory',
    'RecurrentNeuralNetwork',
    # tcn
    'TemporalConvNet',
    # training
    'EarlyStopping',
    'exp_sample_weights',
    # transformer
    'PositionalEncoding',
    'Transformer',
    # rolling
    'CVResult',
    'RollMultiLayerPerceptron',
    '_RollingBasis',
    # loss
    'BaseLoss',
    'DirectionalAccuracyLoss',
    'CalmarLoss',
    'HybridLoss',
    'OmegaLoss',
    'SharpeLoss',
    'SortinoLoss',
]
