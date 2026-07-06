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

.. rubric:: Objective-aligned training

:class:`~fynance.models.objective.ObjectiveModel` trains a network **directly on a
differentiable financial objective** — e.g. :class:`~fynance.models.loss.SharpeLoss`
— rather than MSE against a target. The net outputs *positions* and the loss is
computed on ``positions * returns``: ``fit(X, y)`` reads ``y`` as the realized
returns, and ``predict(X)`` returns positions in ``[-1, 1]``. It is a
``SignalModel``, so it drops straight into a :class:`~fynance.strategy.Strategy`
with an identity signal::

    from fynance.models import ObjectiveModel, SharpeLoss
    from fynance.strategy import Strategy

    model = ObjectiveModel(layers=(16, 8), loss=SharpeLoss(), epochs=60, seed=0)
    strat = Strategy(model=model, signal=lambda positions: positions)

Set ``cost`` (a per-bar proportional fee, e.g. ``0.0026``) to train on the
**net-of-cost** return ``positions * returns - cost * |Δpositions|``: the net then
learns to *hold* rather than churn — the anti-churn lever for high-cost or
high-frequency settings (use the same value as the backtest's
:class:`~fynance.backtest.ProportionalCost`).

Feed it through the research harness via the ``X`` path with ``y`` = returns; see
:doc:`research_workflow`.

**Cross-asset pretraining & persistence.**
:func:`~fynance.models.pretrain_pooled` trains one net on a **pool** of aligned
``(X_i, y_i)`` assets to learn a *shared* signal — each asset stays a contiguous
segment and mini-batches never cross an asset join, so the turnover carry and
temporal order stay intact per asset. The usual workflow then adapts per asset:
:meth:`~fynance.models.objective.ObjectiveModel.clone` a copy with the pretrained
weights and :meth:`~fynance.models.objective.ObjectiveModel.finetune` it on that
asset's own data (``freeze_trunk=True`` trains only the head). A trained model
round-trips to disk with
:meth:`~fynance.models.objective.ObjectiveModel.save` /
:meth:`~fynance.models.objective.ObjectiveModel.load`.

.. rubric:: Distributional (quantile) regression

:class:`~fynance.models.quantile.QuantileModel` trains a feed-forward trunk with
one output per target quantile (default ``taus=(0.1, 0.5, 0.9)``) on
:class:`~fynance.models.loss.PinballLoss`, giving a **distributional** forecast
instead of a single point estimate. Unlike ``ObjectiveModel``, ``fit(X, y)`` reads
``y`` as an ordinary supervised target (e.g. the next-bar return), not a returns
series to combine with positions. It is a ``SignalModel``: ``predict(X)`` returns
the median (or nearest-to-0.5 ``tau``) column, shape ``(T,)``; the full band is
available through
:meth:`~fynance.models.quantile.QuantileModel.predict_quantiles`, shape
``(T, n_taus)``. Quantile columns are trained independently (no crossing penalty),
so non-crossing is enforced **at predict time** by sorting along the quantile
axis::

    from fynance.models import QuantileModel

    model = QuantileModel(taus=(0.1, 0.5, 0.9), layers=(16, 8), epochs=200, seed=0)
    model.fit(X, y)
    point = model.predict(X)              # (T,) median column
    q10, q50, q90 = model.predict_quantiles(X).T

.. rubric:: Regime-conditioned architecture

:class:`~fynance.models.regime_model.RegimeMoE` conditions an objective-aligned
network on the **causal** market regime (:class:`~fynance.features.RegimeDetector`):
the prediction depends on which volatility regime the market is in. ``routing="soft"``
(default) concatenates a learned regime **embedding** to the features through a
shared trunk; ``routing="hard"`` uses one **expert** per regime. The regime label
is produced by a detector fit on the **training** slice only and assigned online,
from a designated positive price/level column of ``X`` (``regime_col``). It reuses
``ObjectiveModel`` for training, so it is a ``SignalModel`` like the above.

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
   objective.pretrain_pooled
   quantile.QuantileModel
   regime_model.RegimeMoE
