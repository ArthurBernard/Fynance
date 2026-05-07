*************************
Recurrent neural networks
*************************

This page documents two categories of recurrent objects:

- **Composable cells** — raw recurrent units with no output projection, designed
  to be embedded inside larger architectures (TCN, Transformers, encoder-decoders).
  They expose only a ``forward`` method; calling ``train_on`` or ``predict``
  raises ``NotImplementedError``.

- **Complete models** — cells wrapped with an output projection layer and the full
  :class:`~fynance.models._base.BaseNeuralNet` training API (``set_optimizer``,
  ``train_on``, ``predict``). Use these directly for walk-forward financial
  forecasting.

The three architectures follow a complexity ladder: vanilla Elman
:class:`~fynance.models.rnn.RecurrentNeuralNetwork` → gated
:class:`~fynance.models.gru.GatedRecurrentUnit` (reset + update gates) →
:class:`~fynance.models.lstm.LongShortTermMemory` (explicit cell state for long
dependencies).

.. currentmodule:: fynance.models

.. rubric:: Composable cells

Raw GRU and LSTM cells without output projection. Pass them as sub-modules to
build custom architectures.

.. autosummary::
   :toctree: generated/

   gru.GRUCell
   lstm.LSTMCell

.. rubric:: Complete models

Ready-to-train models with output projection and the full
:class:`~fynance.models._base.BaseNeuralNet` API.

.. autosummary::
   :toctree: generated/

   rnn.RecurrentNeuralNetwork
   gru.GatedRecurrentUnit
   lstm.LongShortTermMemory
