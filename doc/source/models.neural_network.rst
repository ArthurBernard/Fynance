*********************
Neural network models
*********************

:class:`~fynance.models._base.BaseNeuralNet` is the root of all PyTorch models in
this package. It wraps ``torch.nn.Module`` with higher-level training, prediction
and serialization helpers — :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`,
:meth:`~fynance.models._base.BaseNeuralNet.train_on`,
:meth:`~fynance.models._base.BaseNeuralNet.predict`,
:meth:`~fynance.models._base.BaseNeuralNet.set_data`,
:meth:`~fynance.models._base.BaseNeuralNet.save_model` /
:meth:`~fynance.models._base.BaseNeuralNet.load_model` — so that subclasses only
need to implement ``forward``.

:class:`~fynance.models.mlp.MultiLayerPerceptron` is the feed-forward specialization:
a configurable stack of ``Linear → Dropout → Activation`` blocks, best suited for
tabular or sliding-window features (technical indicators, volatility signals). For
time-ordered sequences, prefer the recurrent architectures described in
:doc:`models.recurrent_neural_network`.

.. rubric:: Inheritance

``BaseNeuralNet`` → ``MultiLayerPerceptron``

``BaseNeuralNet`` → ``_RecurrentBase`` → ``RecurrentNeuralNetwork`` / ``GRUCell``
/ ``LSTMCell`` (see :doc:`models.recurrent_neural_network`)

.. rubric:: Classes

.. currentmodule:: fynance.models

.. autosummary::
   :toctree: generated/

   _base.BaseNeuralNet
   mlp.MultiLayerPerceptron
   objective.ObjectiveModel
