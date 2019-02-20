#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 

Some models, tools, etc for neural network

.. automodule:: fynance.neural_networks
   :members:

"""
from . import roll_neural_network
from .roll_neural_network import RollNeuralNet
from . import roll_multi_neural_networks
from .roll_multi_neural_networks import RollMultiNeuralNet
from . import roll_aggregated_multi_neural_networks
from .roll_aggregated_multi_neural_networks import RollAggrMultiNeuralNet
from . import roll_multi_roll_neural_networks
from .roll_multi_roll_neural_networks import RollMultiRollNeuralNet
#from . import set_neuralnetwork_tools
#from .set_neuralnetwork_tools import *

__all__ = ['RollNeuralNet']
__all__ += ['RollMultiNeuralNet']
__all__ += ['RollAggrMultiNeuralNet']
__all__ += ['RollMultiRollNeuralNet']
# TODO : dev setter function
#__all__ += set_neuralnetwork_tools.__all__