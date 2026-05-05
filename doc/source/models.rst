------------------------------------------
 Financial models (:mod:`fynance.models`)
------------------------------------------

Several financial models — deep learning, econometric and statistical
architectures.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 0
   :padding: 0

   .. grid-item-card:: :octicon:`broadcast;1.2em;sd-mr-1` Attention
      :link: models.attention
      :link-type: doc

      Scaled dot-product and multi-head attention modules for
      Transformer-based architectures.

   .. grid-item-card:: :octicon:`pulse;1.2em;sd-mr-1` Econometric models
      :link: models.econometric_models
      :link-type: doc

      Time-series models: MA, ARMA, ARMA-GARCH, ARMAX-GARCH.

   .. grid-item-card:: :octicon:`workflow;1.2em;sd-mr-1` Neural network models
      :link: models.neural_network
      :link-type: doc

      Multi-layer perceptron and base class for PyTorch neural network
      models.

   .. grid-item-card:: :octicon:`sync;1.2em;sd-mr-1` Recurrent neural networks
      :link: models.recurrent_neural_network
      :link-type: doc

      RNN, GRU and LSTM models with walk-forward training support.

   .. grid-item-card:: :octicon:`history;1.2em;sd-mr-1` Rolling models
      :link: models.rolling
      :link-type: doc

      Walk-forward evaluation wrappers for time-series models.

.. toctree::
   :maxdepth: 1
   :hidden:

   models.attention
   models.econometric_models
   models.neural_network
   models.recurrent_neural_network
   models.rolling
