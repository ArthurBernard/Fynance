#!/usr/bin/env python3
# coding: utf-8

""" Base classes for PyTorch neural network models.

Defines :class:`BaseNeuralNet`, a thin wrapper around
``torch.nn.Module`` that adds higher-level training, prediction and
serialization helpers, and the internal :func:`_type_convert` utility.

These classes are the foundation for all models in :mod:`fynance.models`:
the feed-forward :class:`~fynance.models.mlp.MultiLayerPerceptron`, the
recurrent variants in :mod:`fynance.models.rnn`,
:mod:`fynance.models.gru`, :mod:`fynance.models.lstm`, and the
walk-forward wrappers in :mod:`fynance.models.rolling`.

Main entry points
-----------------
- :class:`BaseNeuralNet` — base class with ``set_optimizer``,
  ``train_on``, ``predict``, ``set_data``, ``save_model``,
  ``load_model`` helpers.

"""

from __future__ import annotations

# Built-in packages
# Third-party packages
import numpy as np
import polars as pl
import torch
import torch.nn
from numpy.typing import NDArray

__all__ = ['BaseNeuralNet']


_TYPE_HANDLER = {
    "int": torch.int64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "float": torch.float64,
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
}


class BaseNeuralNet(torch.nn.Module):
    """ Base object for neural network model with PyTorch.

    Thin wrapper around ``torch.nn.Module`` that bundles the boilerplate
    of training a financial model: criterion + optimizer setup, a
    one-batch ``train_on`` step, gradient-free ``predict``, data
    coercion from NumPy / polars / tensor, and weight serialization.
    Subclass it (or one of the higher-level subclasses such as
    :class:`~fynance.models.mlp.MultiLayerPerceptron`) and implement the
    ``forward`` method to define a new architecture; everything else is
    inherited.

    Inherits of torch.nn.Module object with some higher level methods.

    **Public API contract (stable for the 1.x series)**

    - **Shapes** — feed-forward subclasses (e.g.
      :class:`~fynance.models.mlp.MultiLayerPerceptron`)
      expect ``X`` of shape ``(T, N)`` and ``y`` of shape ``(T, M)``,
      where ``T`` is the number of observations, ``N`` the number of
      input features and ``M`` the number of targets. Recurrent
      subclasses in :mod:`fynance.models.rnn`,
      :mod:`fynance.models.gru`, :mod:`fynance.models.lstm`
      consume ``X`` of shape ``(T, S, N)``, where ``S`` is the sequence
      length.
    - **Dtypes** — :meth:`set_data` coerces inputs through
      :meth:`_set_data`. The expected default for both ``X`` and ``y``
      is a floating tensor (``torch.float32`` is the de-facto convention
      in the project doctests). Pass ``x_type`` / ``y_type`` explicitly
      to override; mismatched dtypes between ``X``, ``y`` and the model
      parameters will raise at the first ``forward`` pass.
    - **Device** — the wrapper does **not** move tensors automatically.
      Models live on CPU unless the caller explicitly calls ``.to(device)``
      on both the module and the data tensors before training.
    - **State invariants** — typical lifecycle:
      :meth:`set_optimizer` → :meth:`set_data` (optional) →
      :meth:`train_on` (loops) → :meth:`predict`. ``train_on`` requires
      ``criterion`` and ``optimizer`` to be set; calling it before
      :meth:`set_optimizer` raises ``AttributeError``.
    - **Serialization** — :meth:`save_model` / :meth:`load_model`
      persist the module ``state_dict`` and, when ``save_optimizer`` /
      ``load_optimizer`` is True, the optimizer ``state_dict``. Random
      seeds, learning-rate schedulers and the cached training data
      (``self.X``, ``self.y``) are **not** serialized.

    Attributes
    ----------
    criterion : torch.nn.modules.loss.Loss
        A loss function.
    optimizer : torch.optim.Optimizer
        An optimizer algorithm.
    N, M : int
        Respectively input and output dimension.

    Methods
    -------
    set_optimizer
    train_on
    predict
    set_data
    save_model
    load_model

    See Also
    --------
    fynance.models.mlp.MultiLayerPerceptron,
    fynance.models.rolling.RollMultiLayerPerceptron

    """

    lr_scheduler = None
    optimizer = None
    seed_torch = None
    seed_numpy = None

    def __init__(self):
        """ Initialize. """
        torch.nn.Module.__init__(self)

    def set_optimizer(self, criterion, optimizer, params=None, **kwargs):
        """ Set the optimizer object.

        Set optimizer object with specified `criterion` as loss function and
        any `kwargs` as optional parameters.

        Parameters
        ----------
        criterion : Callabletorch.nn.modules.loss
            A loss function.
        optimizer : torch.optim.Optimizer
            An optimizer algorithm.
        params : object or iterable object
            Layer of parameters to optimize or dicts defining parameter groups.
            If set to None then all parameters of model will be optimized.
            Default is None.
        **kwargs
            Keyword arguments of ``optimizer``, cf PyTorch documentation [1]_.

        Returns
        -------
        BaseNeuralNet
            Self object model.

        References
        ----------
        .. [1] https://pytorch.org/docs/stable/optim.html

        """
        if params is None:
            params = self.parameters()

        elif isinstance(params, list):
            params = [{'params': p.parameters()} for p in params]

        else:
            params = params.parameters()

        self.criterion = criterion()
        self.optimizer = optimizer(params, **kwargs)

        return self

    def set_lr_scheduler(self, lr_scheduler, **kwargs):
        """ Set dynamic learning rate.

        Parameters
        ----------
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            Method from ``torch.optim.lr_scheduler`` to wrap
            ``self.optimizer``, cf module ``torch.optim.lr_scheduler`` in
            PyTorch documentation [2]_.
        **kwargs
            Keyword arguments to pass to the learning rate scheduler.

        References
        ----------
        .. [2] https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

        """
        if self.optimizer:
            self.lr_scheduler = lr_scheduler(self.optimizer, **kwargs)

        else:
            raise ValueError('You should specify an optimizer object, '
                             'see `set_optimizer` method.')

        return self

    @torch.enable_grad()
    def train_on(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Trains the neural network model on a single batch.

        Runs one forward / backward / optimizer-step cycle on the batch
        ``(X, y)``. As a side effect, gradients of all parameters are
        zeroed before the forward pass and the optimizer state is
        advanced afterwards. If a learning-rate scheduler has been
        registered via :meth:`set_lr_scheduler`, its ``step`` is also
        called.

        Parameters
        ----------
        X, y : torch.Tensor
            Respectively inputs and outputs to train model. Shapes must
            match what ``self.forward`` expects (see the class-level
            "Public API contract" section).

        Returns
        -------
        torch.Tensor
            The loss tensor produced by ``self.criterion(self(X), y)``,
            with gradient already consumed by ``loss.backward()``.

        Raises
        ------
        AttributeError
            If :meth:`set_optimizer` has not been called yet.

        """
        self.optimizer.zero_grad()  # type: ignore[attr-defined]
        outputs = self(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()  # type: ignore[attr-defined]

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """ Predicts outputs of neural network model.

        Runs ``self.forward(X)`` under :func:`torch.no_grad`, so no
        autograd graph is built. The returned tensor is detached and
        lives on the same device as the model parameters.

        Parameters
        ----------
        X : torch.Tensor
           Inputs to compute prediction. Same shape and dtype contract
           as :meth:`train_on`.

        Returns
        -------
        torch.Tensor
           Outputs prediction (detached, gradient-free).

        """
        return self(X).detach()

    def set_data(self, X: NDArray | torch.Tensor | pl.DataFrame, y: NDArray | torch.Tensor | pl.DataFrame, x_type=None, y_type=None):
        """ Set data inputs and outputs.

        Coerces ``X`` and ``y`` to :class:`torch.Tensor` and caches them
        as ``self.X`` / ``self.y``. After the call the attributes
        ``self.T`` (number of observations), ``self.N`` (input columns)
        and ``self.M`` (output columns) are set.

        Parameters
        ----------
        X, y : array-like
            Respectively input and output data. Accepted types:
            :class:`numpy.ndarray`, :class:`torch.Tensor`,
            :class:`polars.DataFrame`. Shapes must be ``(T, N)`` and
            ``(T, M)`` respectively.
        x_type, y_type : torch.dtype, optional
            Target dtypes for the resulting tensors. Default is `None`,
            which preserves the input dtype.

        Returns
        -------
        BaseNeuralNet
            ``self``, to allow chaining.

        Raises
        ------
        ValueError
            If ``self.N`` / ``self.M`` were already set and ``X`` / ``y``
            do not match, or if ``X`` and ``y`` have different lengths.

        """
        if hasattr(self, 'N') and self.N != X.shape[1]:  # type: ignore[has-type]
            raise ValueError('X must have {} input columns'.format(self.N))  # type: ignore[has-type]

        if hasattr(self, 'M') and self.M != y.shape[1]:  # type: ignore[has-type]
            raise ValueError('y must have {} output columns'.format(self.M))  # type: ignore[has-type]

        self.X = self._set_data(X, dtype=x_type)
        self.y = self._set_data(y, dtype=y_type)
        self.T, self.N = self.X.size()
        T_veri, self.M = self.y.size()

        if self.T != T_veri:
            raise ValueError('{} time periods in X differents of {} time \
                             periods in y'.format(self.T, T_veri))

        return self

    def set_seed(self, seed_torch=None, seed_numpy=None):
        r""" Set seed for PyTorch and NumPy random number generator.

        Parameters
        ----------
        seed_torch, seed_numpy : bool or int, optional
            If `seed` is an int :math:`0 < seed < 2^32` set respectively
            PyTorch and NumPy seed with the number. Otherwise if is True
            then choose a random number, else doesn't set seed.

        """
        self.seed_torch = self._set_seed(seed_torch)
        self.seed_numpy = self._set_seed(seed_numpy)
        torch.manual_seed(self.seed_torch)
        np.random.seed(self.seed_numpy)

    def _set_seed(self, seed):
        if isinstance(seed, int) and 0 <= seed < 2 ** 32:

            return seed

        return np.random.randint(0, 2 ** 32)

    def _set_data(self, X, dtype=None):
        """ Convert array-like data to tensor. """
        if isinstance(X, np.ndarray):

            return torch.from_numpy(X)

        elif isinstance(X, pl.DataFrame):
            return torch.from_numpy(X.to_numpy())

        elif isinstance(X, torch.Tensor):

            return X

        else:
            raise ValueError('Unkwnown data type: {}'.format(type(X)))

    def save_model(self, path, save_optimizer=False):
        """ Save the model with this weights and parameters.

        Parameters
        ----------
        path : str or os.PathLike object
            Path to save the model.
        save_optimizer : bool, optional
            If True, then save also the optimizer.

        """
        state_dict = {"model": self.state_dict()}
        if save_optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()

        torch.save(state_dict, path)

    def load_model(self, path, load_optimizer=False):
        """ Save the model with this weights and parameters.

        Parameters
        ----------
        path : str or os.PathLike object
            Path to load the model.
        load_optimizer : bool, optional
            If True, then load also the optimizer.

        """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict['model'])

        if load_optimizer:
            if 'optimizer' not in state_dict:
                raise ValueError('No optimizer available, set `load_optimizer`'
                                 ' to False')

            elif getattr(self, 'optimizer', None) is None:
                raise ValueError('You should specify an optimizer object, '
                                 'see `set_optimizer` method.')

            self.optimizer.load_state_dict(state_dict['optimizer'])


def _type_convert(dtype):
    if dtype is np.float64:
        return torch.float64

    elif dtype is np.float32:
        return torch.float32

    elif dtype is np.float16:
        return torch.float16

    elif dtype is np.uint8:
        return torch.uint8

    elif dtype is np.int8:
        return torch.int8

    elif dtype is np.int16:
        return torch.int16

    elif dtype is np.int32:
        return torch.int32

    elif dtype is np.int64:
        return torch.int64

    else:
        raise ValueError('Unkwnown type: {}'.format(str(dtype)))
