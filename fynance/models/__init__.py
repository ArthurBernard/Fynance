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

    models.attention
    models.econometric_models
    models.neural_network
    models.recurrent_neural_network
    models.rolling

"""

from . import (
    attention,
    econometric_models,
    econometric_models_cy,
    neural_network,
    recurrent_neural_network,
    rolling,
)
from .attention import MultiHeadAttention, ScaledDotProductAttention
from .econometric_models import ARMA, ARMA_GARCH, ARMAX_GARCH, MA, get_parameters
from .econometric_models_cy import (
    ARMA_cy,
    ARMA_GARCH_cy,
    ARMAX_GARCH_cy,
    MA_cy,
    get_parameters_cy,
)
from .neural_network import BaseNeuralNet, MultiLayerPerceptron
from .recurrent_neural_network import (
    GatedRecurrentUnit,
    LongShortTermMemory,
    RecurrentNeuralNetwork,
)
from .rolling import RollMultiLayerPerceptron, _RollingBasis

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
    # neural_network
    'BaseNeuralNet',
    'MultiLayerPerceptron',
    # recurrent_neural_network
    'GatedRecurrentUnit',
    'LongShortTermMemory',
    'RecurrentNeuralNetwork',
    # rolling
    'RollMultiLayerPerceptron',
    '_RollingBasis',
]
