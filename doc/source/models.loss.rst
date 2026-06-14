**************
Loss functions
**************

Differentiable financial loss functions for PyTorch training. They are drop-in PyTorch criterions usable with :meth:`~fynance.models._base.BaseNeuralNet.set_optimizer`.

.. currentmodule:: fynance.models.loss

.. autosummary::
   :toctree: generated/

   BaseLoss
   SharpeLoss
   SortinoLoss
   DirectionalAccuracyLoss
   CalmarLoss
   OmegaLoss
   HybridLoss
