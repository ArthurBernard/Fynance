#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predictive-uncertainty wrappers around SignalModel-conforming nets.

Two complementary ways to turn a point-prediction net into a model that also
reports how *uncertain* it is about that prediction, both exposing the same
extra API (``predict_std``) on top of the usual ``SignalModel`` ``fit`` /
``predict``:

- :class:`DeepEnsemble` trains several independently initialized copies of a
  net and uses their disagreement as the uncertainty signal (Lakshminarayanan
  et al., 2017).
- :class:`MCDropout` wraps a *single* dropout-bearing net and keeps dropout
  active at inference, averaging several stochastic forward passes (Gal &
  Ghahramani, 2016, "Dropout as a Bayesian Approximation").

Both report an **epistemic** uncertainty proxy (how much the fit itself could
plausibly have differed), not the aleatoric noise inherent to the data. Neither
changes the training objective: they wrap an already-trained (or trainable)
``SignalModel`` and add ``predict_std`` / ``predict_members`` on top.

"""

from __future__ import annotations

# Built-in packages
import warnings
from typing import Any, Callable

# Third-party packages
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.modules.dropout import _DropoutNd

__all__ = ['DeepEnsemble', 'MCDropout']


def _flatten(pred: Any) -> NDArray:
    """ Coerce a model prediction (tensor or array-like) to a flat 1-D array. """
    arr = pred.numpy() if hasattr(pred, 'numpy') else np.asarray(pred)

    return np.asarray(arr, dtype=np.float64).reshape(-1)


class DeepEnsemble:
    r""" Epistemic-uncertainty proxy from an ensemble of independent nets.

    Trains ``n_members`` independent copies of a ``SignalModel``-conforming net
    (built fresh by ``factory`` for each member) and aggregates their
    predictions. The **member-to-member spread** — :meth:`predict_std` — is a
    proxy for *epistemic* uncertainty (how much the fit itself could have
    differed had training gone slightly differently), as distinct from the
    *aleatoric* noise in the data itself. See Lakshminarayanan, Pritzel & Blundell
    (2017), "Simple and Scalable Predictive Uncertainty Estimation using Deep
    Ensembles".

    Conforms to the :class:`~fynance.core.protocols.SignalModel` protocol
    (``fit``/``predict``): it drops into the harness like any other model, with
    :meth:`predict_std` / :meth:`predict_members` as the extra uncertainty API.

    Parameters
    ----------
    factory : callable
        No-argument callable returning a **fresh**, unfit
        ``SignalModel``-conforming net (e.g. a closure building a
        :class:`~fynance.models.mlp.MultiLayerPerceptron` and calling
        ``set_optimizer`` on it, or any object with ``fit(X, y)`` /
        ``predict(X)``). Called once per member, per :meth:`fit` call.
    n_members : int
        Number of independently trained members. Default 5.
    seed : int
        Base seed. Member ``i`` is built and trained under
        ``torch.manual_seed(seed + i)`` (and ``np.random.seed(seed + i)``), so
        the ensemble as a whole is reproducible while members differ from one
        another.

    Notes
    -----
    **Determinism limits.** :meth:`fit` reseeds the *global* PyTorch and NumPy
    generators immediately before building and training each member. This
    reproducibly diversifies members whose randomness (parameter
    initialization, dropout masks, data shuffling) is drawn from those global
    generators — which covers :class:`~fynance.models.mlp.MultiLayerPerceptron`
    and the other :class:`~fynance.models._base.BaseNeuralNet` subclasses. A
    member that instead draws from its **own private** ``torch.Generator``
    (for example :class:`~fynance.models.objective.ObjectiveModel`'s
    mini-batch shuffling, seeded from its own constructor ``seed`` argument
    rather than the global RNG) is unaffected by this reseed step: give such a
    ``factory`` a distinct explicit per-member seed if independent draws from
    that private generator are also required. Two full :meth:`fit` runs with
    the same ``DeepEnsemble.seed`` (and the same ``factory``/data) are
    otherwise reproducible.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from fynance.models.mlp import MultiLayerPerceptron
    >>> from fynance.models.uncertainty import DeepEnsemble
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((40, 2)).astype(np.float32)
    >>> y = np.sin(X[:, :1]).astype(np.float32)
    >>> def factory():
    ...     net = MultiLayerPerceptron(2, 1, layers=[8], activation=torch.nn.Tanh)
    ...     net.set_optimizer(torch.nn.MSELoss, torch.optim.Adam, lr=1e-2)
    ...     return net
    >>> ens = DeepEnsemble(factory, n_members=3, seed=0).fit(X, y)
    >>> ens.predict(X).shape
    (40,)
    >>> ens.predict_members(X).shape
    (3, 40)

    See Also
    --------
    fynance.models.uncertainty.MCDropout,
    fynance.models.ensemble.StackingEnsemble

    """

    def __init__(
        self,
        factory: Callable[[], Any],
        n_members: int = 5,
        seed: int = 0,
    ):
        self.factory = factory
        self.n_members = n_members
        self.seed = seed
        self.members: list[Any] = []

    def fit(self, X: NDArray, y: NDArray) -> DeepEnsemble:
        """ Build and train ``n_members`` independent copies of ``factory()``.

        Parameters
        ----------
        X, y : array-like
            Training data forwarded verbatim to each member's ``fit(X, y)``.

        Returns
        -------
        DeepEnsemble
            ``self``.

        """
        self.members = []
        for i in range(self.n_members):
            seed_i = self.seed + i
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            member = self.factory()
            member.fit(X, y)
            self.members.append(member)

        return self

    def predict_members(self, X: NDArray) -> NDArray:
        """ Per-member predictions, shape ``(n_members, T)``.

        Parameters
        ----------
        X : array-like
            Inputs, shape ``(T, N)``.

        Returns
        -------
        numpy.ndarray
            Stacked per-member predictions, shape ``(n_members, T)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        """
        if not self.members:

            raise RuntimeError('DeepEnsemble must be fit before predict')

        return np.stack([_flatten(member.predict(X)) for member in self.members])

    def predict(self, X: NDArray) -> NDArray:
        """ Ensemble mean prediction, shape ``(T,)``. """

        return self.predict_members(X).mean(axis=0)

    def predict_std(self, X: NDArray) -> NDArray:
        """ Cross-member standard deviation, shape ``(T,)`` — epistemic proxy. """

        return self.predict_members(X).std(axis=0)


class MCDropout:
    r""" Epistemic-uncertainty proxy via Monte Carlo Dropout at inference.

    Wraps **one** dropout-bearing :class:`torch.nn.Module` (typically a
    :class:`~fynance.models._base.BaseNeuralNet` subclass such as
    :class:`~fynance.models.mlp.MultiLayerPerceptron` built with ``drop > 0``)
    and approximates a Bayesian posterior by keeping dropout **active at
    inference** and averaging ``n_samples`` stochastic forward passes (Gal &
    Ghahramani, 2016, "Dropout as a Bayesian Approximation"). The spread across
    those passes — :meth:`predict_std` — is the epistemic-uncertainty proxy.

    Technique
    ---------
    A plain ``model.eval()`` freezes dropout too (no stochasticity left to
    sample from). :meth:`predict` / :meth:`predict_std` instead first put the
    *whole* model in eval mode (so batch-norm and similar layers behave as at
    inference) and then explicitly switch back to train mode **only the
    ``torch.nn.Dropout*`` submodules**, so dropout masks are still resampled on
    every forward pass while everything else stays deterministic. The model's
    original mode is restored before returning (or raising).

    Conforms to the :class:`~fynance.core.protocols.SignalModel` protocol
    (``fit``/``predict``).

    Parameters
    ----------
    model : torch.nn.Module
        A ``SignalModel``-conforming net (``fit(X, y)`` / ``predict(X)``) with
        at least one ``torch.nn.Dropout*`` submodule. A model with **no**
        dropout submodule still works but is pointless (every one of the
        ``n_samples`` passes is then identical, so :meth:`predict_std` is
        ``~0``): a ``UserWarning`` is raised at construction in that case.
    n_samples : int
        Number of stochastic forward passes averaged by :meth:`predict` /
        :meth:`predict_std`. Default 50.
    seed : int
        Seed (``torch.manual_seed``) applied before drawing the ``n_samples``
        dropout masks, so both methods are reproducible for a given instance.

    Notes
    -----
    Increasing ``n_samples`` reduces the variance of the **mean** estimate
    returned by :meth:`predict` (the mean of ``n`` i.i.d. stochastic passes has
    variance :math:`\sigma^2 / n` by the law of large numbers) but does not
    shrink :meth:`predict_std` itself, which converges to the model's
    intrinsic dropout-induced spread rather than to zero.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from fynance.models.mlp import MultiLayerPerceptron
    >>> from fynance.models.uncertainty import MCDropout
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((40, 2)).astype(np.float32)
    >>> y = np.sin(X[:, :1]).astype(np.float32)
    >>> net = MultiLayerPerceptron(2, 1, layers=[8], activation=torch.nn.Tanh,
    ...                            drop=0.2)
    >>> _ = net.set_optimizer(torch.nn.MSELoss, torch.optim.Adam, lr=1e-2)
    >>> mc = MCDropout(net, n_samples=10, seed=0).fit(X, y)
    >>> mc.predict(X).shape
    (40,)
    >>> bool((mc.predict_std(X) >= 0).all())
    True

    See Also
    --------
    fynance.models.uncertainty.DeepEnsemble

    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 50,
        seed: int = 0,
    ):
        self.model = model
        self.n_samples = n_samples
        self.seed = seed
        if not any(isinstance(m, _DropoutNd) for m in model.modules()):
            warnings.warn(
                'MCDropout wraps a model with no torch.nn.Dropout* submodule; '
                'every stochastic pass will be identical so predict_std will '
                'be ~0. Build the wrapped model with a nonzero `drop` '
                'probability for a meaningful uncertainty estimate.',
                stacklevel=2,
            )

    def fit(self, X: NDArray, y: NDArray) -> MCDropout:
        """ Fit the wrapped model normally (dropout trains as usual).

        Parameters
        ----------
        X, y : array-like
            Training data forwarded to the wrapped model's ``fit(X, y)``.

        Returns
        -------
        MCDropout
            ``self``.

        """
        self.model.fit(X, y)  # type: ignore[operator]

        return self

    def _to_tensor(self, X: NDArray | torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            tensor = X

        else:
            tensor = torch.from_numpy(np.asarray(X)).clone()

        if tensor.is_floating_point() and tensor.dtype != torch.get_default_dtype():

            return tensor.to(torch.get_default_dtype())

        return tensor

    def _samples(self, X: NDArray) -> NDArray:
        """ Stack ``n_samples`` stochastic forward passes, shape ``(n_samples, T)``. """
        Xt = self._to_tensor(X)

        was_training = self.model.training
        self.model.eval()
        dropout_modules = [m for m in self.model.modules()
                          if isinstance(m, _DropoutNd)]
        for module in dropout_modules:
            module.train()

        torch.manual_seed(self.seed)
        samples = []
        try:
            with torch.no_grad():
                for _ in range(self.n_samples):
                    out = self.model(Xt)
                    samples.append(_flatten(out))

        finally:
            self.model.train(was_training)

        return np.stack(samples)

    def predict(self, X: NDArray) -> NDArray:
        """ Mean over ``n_samples`` stochastic forward passes, shape ``(T,)``. """

        return self._samples(X).mean(axis=0)

    def predict_std(self, X: NDArray) -> NDArray:
        """ Std over ``n_samples`` stochastic forward passes — epistemic proxy. """

        return self._samples(X).std(axis=0)
