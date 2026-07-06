------------------------------------------
 Financial models (:mod:`fynance.models`)
------------------------------------------

Several financial models — deep learning, econometric and statistical
architectures.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 0
   :padding: 0

   .. grid-item-card:: :octicon:`eye;1.2em;sd-mr-1` Attention
      :link: models.attention
      :link-type: doc

      Scaled dot-product and multi-head attention modules for
      Transformer-based architectures.

   .. grid-item-card:: :octicon:`law;1.2em;sd-mr-1` Econometric models
      :link: models.econometric_models
      :link-type: doc

      Time-series models: MA, ARMA, ARMA-GARCH, ARMAX-GARCH.

   .. grid-item-card:: :octicon:`dependabot;1.2em;sd-mr-1` Neural network models
      :link: models.neural_network
      :link-type: doc

      Multi-layer perceptron and base class for PyTorch neural network
      models.

   .. grid-item-card:: :octicon:`iterations;1.2em;sd-mr-1` Recurrent neural networks
      :link: models.recurrent_neural_network
      :link-type: doc

      RNN, GRU and LSTM models with walk-forward training support.

   .. grid-item-card:: :octicon:`stack;1.2em;sd-mr-1` Temporal Convolutional Network
      :link: models.tcn
      :link-type: doc

      Causal dilated convolutional network for sequences.

   .. grid-item-card:: :octicon:`eye;1.2em;sd-mr-1` Transformer
      :link: models.transformer
      :link-type: doc

      Causal Transformer encoder with positional encoding.

   .. grid-item-card:: :octicon:`git-merge;1.2em;sd-mr-1` Ensemble
      :link: models.ensemble
      :link-type: doc

      Direction + magnitude stacking with an out-of-fold meta-model.

   .. grid-item-card:: :octicon:`graph;1.2em;sd-mr-1` Uncertainty
      :link: models.uncertainty
      :link-type: doc

      Deep ensembles and MC Dropout predictive-uncertainty wrappers.

   .. grid-item-card:: :octicon:`flame;1.2em;sd-mr-1` Loss functions
      :link: models.loss
      :link-type: doc

      Differentiable Sharpe/Sortino/Calmar/Omega/directional losses.

   .. grid-item-card:: :octicon:`gear;1.2em;sd-mr-1` Training utilities
      :link: models.training
      :link-type: doc

      Sample weighting and early stopping.

   .. grid-item-card:: :octicon:`history;1.2em;sd-mr-1` Rolling models
      :link: models.rolling
      :link-type: doc

      Walk-forward evaluation wrappers for time-series models.

   .. grid-item-card:: :octicon:`checklist;1.2em;sd-mr-1` Purged walk-forward tuning
      :link: models.tuning
      :link-type: doc

      Grid/random hyperparameter search scored on purged walk-forward folds.

.. toctree::
   :maxdepth: 1
   :hidden:

   models.attention
   models.econometric_models
   models.ensemble
   models.loss
   models.neural_network
   models.recurrent_neural_network
   models.rolling
   models.tcn
   models.training
   models.transformer
   models.tuning
   models.uncertainty
