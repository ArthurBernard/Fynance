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
    models.tuning

"""

from . import (
    _base,
    _recurrent_base,
    attention,
    conformal,
    econometric_models,
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
    tuning,
)
from ._base import BaseNeuralNet
from .attention import MultiHeadAttention, ScaledDotProductAttention
from .conformal import ConformalWrapper, rolling_conformal
from .econometric_models import ARMA, ARMA_GARCH, ARMAX_GARCH, MA, get_parameters
from .ensemble import StackingEnsemble
from .gru import GatedRecurrentUnit, GRUCell
from .loss import (
    BaseLoss,
    CalmarLoss,
    DirectionalAccuracyLoss,
    HybridLoss,
    OmegaLoss,
    PinballLoss,
    SharpeLoss,
    SortinoLoss,
)
from .lstm import LongShortTermMemory, LSTMCell
from .mlp import MultiLayerPerceptron
from .objective import ObjectiveModel, pretrain_pooled
from .quantile import QuantileModel
from .regime_model import RegimeMoE
from .rnn import RecurrentNeuralNetwork
from .rolling import CVResult, RollMultiLayerPerceptron, _RollingBasis
from .tcn import TemporalConvNet
from .training import EarlyStopping, exp_sample_weights
from .transformer import PositionalEncoding, Transformer
from .tuning import SearchResult, walk_forward_search
from .uncertainty import DeepEnsemble, MCDropout

# Frozen public surface for the 1.x series — names listed here are
# guaranteed to remain importable from ``fynance.models`` until the
# next major version. New names may be appended (additive change), but
# nothing in this list will be removed without a deprecation cycle.
__all__ = [
    # attention
    'MultiHeadAttention',
    'ScaledDotProductAttention',
    # conformal
    'ConformalWrapper',
    'rolling_conformal',
    # econometric_models
    'ARMA',
    'ARMA_GARCH',
    'ARMAX_GARCH',
    'MA',
    'get_parameters',
    # _base / mlp
    'BaseNeuralNet',
    'StackingEnsemble',
    'MultiLayerPerceptron',
    # objective-aligned training
    'ObjectiveModel',
    'pretrain_pooled',
    # distributional (quantile) regression
    'QuantileModel',
    # regime-conditioned architecture
    'RegimeMoE',
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
    'PinballLoss',
    'SharpeLoss',
    'SortinoLoss',
    # tuning
    'SearchResult',
    'walk_forward_search',
    # uncertainty
    'DeepEnsemble',
    'MCDropout',
]
