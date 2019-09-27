Rolling models
==============
Some tools/objects to roll a model, currently this object work only with neural network models with PyTorch package. You can find in :mod:`fynance.neural_network` some other rolling neural network working with Keras (Tensorflow or Theano). And you can also find in :mod:`fynance.algorithms.allocation` a function to roll some portfolio allocation algorithms.

.. currentmodule:: fynance.models.rolling

.. autosummary::

   fynance.models.rolling._RollingBasis
   fynance.models.rolling.RollMultiLayerPerceptron

.. autoclass:: _RollingBasis
   :special-members:
   :members:
   :show-inheritance:

.. autoclass:: RollMultiLayerPerceptron
   :special-members:
   :members:
   :show-inheritance: