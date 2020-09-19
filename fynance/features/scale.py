#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-11 18:47:27
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-19 10:54:36

""" Object to scale data. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages
from fynance.features.roll_functions import roll_min, roll_max
from fynance.features.momentums import *

__all__ = ["normalize", "roll_normalize", "roll_standardize", "Scale",
           "standardize"]


# TODO :
#     - Use wrapper for axis for scale methods
#     - Use wrapper for axis for standardize and normalize functions
#     - Finish functions or method to scale with moving functions.


_HANDLER_MOMENTUM = {
    "s": [sma, smstd],
    "w": [wma, wmstd],
    "e": [ema, emstd],
}


def _get_norm_params(X, axis=0):
    params = {
        "m": np.min(X, axis=axis),
        "s": np.max(X, axis=axis),
    }

    return params


def _normalize(X, m, s, a, b):

    return (b - a) * (X - m) / (s - m) + a


def _revert_normalize(X, m, s, a, b):

    return _normalize(X, a, b, m, s)


def _get_std_params(X, axis=0):
    params = {
        "m": np.mean(X, axis=axis),
        "s": np.std(X, axis=axis),
    }

    return params


def _standardize(X, m, s, a, b):

    return b * (X - m) / s + a


def _revert_standardize(X, m, s, a, b):

    return _standardize(X, a, b, m, s)


def _get_roll_norm_params(X, w, axis=0):
    params = {
        "m": roll_min(X, w, axis=axis),
        "s": roll_max(X, w, axis=axis),
    }

    return params


def _get_roll_std_params(X, w, kind_moment="s", axis=0):
    params = {
        "m": _HANDLER_MOMENTUM[kind][0](X, w=w, axis=axis),
        "s": _HANDLER_MOMENTUM[kind][1](X, w=w, axis=axis),
    }

    return params


class Scale:
    """ Object to scale data.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Data to fit the parameters of scale transformation.
    kind : str, optional
        - "std" : Standardized scale transformation (default), see
          :func:`~fynance.features.scale.standardize`.
        - "norm" : Normalized scale transformation, see
          :func:`~fynance.features.scale.normalize`.
        - "raw" : No scale is apply.
        - "roll_std" : Standardized scale transformation, computed with
          rolling mean and standard deviation (see
          :func:`~fynance.features.scale.roll_standardize`).
        - "roll_norm" : Normalized scale transformation, computed with
          roling minimum and maximum (see
          :func:`~fynance.features.scale.roll_normalize`).
    a, b : float or array_like, optional
        Some scale factors to apply after the transformation. By default is
        respectively 0 and 1.
    axis : int, optional
        Axis along which compute the scale parameters. Default is 0.
    **kwargs : keyword arguments for particular functions
        E.g: for rolling function set ``w`` the lagged window (see
        :func:`~fynance.features.scale.roll_normalize`) or for rolling
        standardization set ``kind_moment={"s", "w", "e"}`` (see
        :func:`~fynance.features.scale.roll_standardize`).

    Methods
    -------
    fit
    scale
    revert

    Attributes
    ----------
    func : callable
        The scale function.
    revert_func : callable
        The revert scale function.
    params : dict
        Parameters of the scale transformation.
    axis : int
        The axis along which is computed the scale parameters.
    kind : str
        The kind of scale transformation.

    See Also
    --------
    normalize, standardize, roll_standardize, roll_normalize

    """

    handle_func = {
        "raw": lambda x, a, b: x,
        "norm": _normalize,
        "std": _standardize,
        "roll_norm": _normalize,
        "roll_std": _standardize,
    }
    handle_params = {
        "raw": lambda x: {},
        "norm": _get_norm_params,
        "std": _get_std_params,
        "roll_norm": _get_roll_norm_params,
        "roll_std": _get_roll_std_params,
    }
    handle_revert = {
        "raw": lambda x, a, b: x,
        "norm": _revert_normalize,
        "std": _revert_standardize,
        "roll_norm": _revert_normalize,
        "roll_std": _revert_standardize,
    }

    def __init__(self, X, kind="std", a=0., b=1., axis=0):
        """ Initialize the scale object. """
        self.func = self.handle_func[kind]
        self.revert_func = self.handle_revert[kind]
        self.kind = kind
        self.axis = axis
        self.fit(X, kind, a, b, axis)

    def __call__(self, X, axis=None):
        """ Callable method to scale data with fitted parameters.

        Parameters
        ----------
        X : np.ndarray[dtype, ndim=1 or 2]
            Data to scale.

        Returns
        -------
        np.ndarray[dtype, ndim=1 or 2]
            Scalled data.

        """
        return self.scale(X, axis)

    def __repr__(self):
        """ Return string representation. """
        return ("Scale transformation '{}' with the following parameters: {}"
                "".format(self.kind, self.parmas))

    def fit(self, X, kind=None, a=0., b=1., axis=0, **kwargs):
        """ Compute the parameters of the scale transformation.

        Parameters
        ----------
        X : np.ndarray[dtype, ndim=1 or 2]
            Data to fit the parameters of scale transformation.
        kind : str, optional
            - "std" : Standardized scale transformation (default), see
              :func:`~fynance.features.scale.standardize`.
            - "norm" : Normalized scale transformation, see
              :func:`~fynance.features.scale.normalize`.
            - "raw" : No scale is apply.
            - "roll_std" : Standardized scale transformation, computed with
              rolling mean and standard deviation (see
              :func:`~fynance.features.scale.roll_standardize`).
            - "roll_norm" : Normalized scale transformation, computed with
              roling minimum and maximum (see
              :func:`~fynance.features.scale.roll_normalize`).
        a, b : float or array_like, optional
            Some scale factors to apply after the transformation. By default is
            respectively 0 and 1.
        axis : int, optional
            Axis along which compute the scale parameters. Default is 0.
        **kwargs : keyword arguments for particular functions
            E.g: for rolling function set ``w`` the lagged window (see
            :func:`~fynance.features.scale.roll_normalize`) or for rolling
            standardization set ``kind_moment={"s", "w", "e"}`` (see
            :func:`~fynance.features.scale.roll_standardize`).

        """
        if kind is None:
            kind = self.kind

        if axis is None:
            axis = self.axis

        self.params = self.handle_params[kind](X, axis=axis, **kwargs)
        self.params.update({"a": a, "b": b})

    def scale(self, X, axis=None):
        """ Scale the data with the fitted parameters.

        Parameters
        ----------
        X : np.ndarray[dtype, ndim=1 or 2]
            Data to scale.

        Returns
        -------
        np.ndarray[dtype, ndim=1 or 2]
            Scalled data.

        """
        if axis is None:
            axis = self.axis

        if axis == 1:
            self.func(X.T, **self.params).T

        return self.func(X, **self.params)

    def revert(self, X, axis=None):
        """ Revert the transformation of the scale with the fitted parameters.

        Parameters
        ----------
        X : np.ndarray[dtype, ndim=1 or 2]
            Data to revert the scale.

        Returns
        -------
        np.ndarray[dtype, ndim=1 or 2]
            The revert transformed data.

        """
        if axis is None:
            axis = self.axis

        if axis == 1:
            self.revert_func(X.T, **self.params).T

        return self.revert_func(X, **self.params)


def standardize(X, a=0, b=1, axis=0):
    r""" Substitutes the mean and divid by the standard deviation.

    .. math::

        Standardize(X) = b \times \frac{X - X_{mean}}{X_{std}} + a

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Data to scale.
    a, b : float or array_like, optional
        Respectively an additional and multiply factor.
    axis : int, optional
        Axis along which to scale the data.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        The scaled data.

    See Also
    --------
    Scale, normalize, roll_standardize

    """
    m = np.mean(X, axis=axis)
    s = np.std(X, axis=axis)

    if axis == 1:

        return _standardize(X.T, m, s, a, b).T

    return _standardize(X, m, s, a, b)


def roll_standardize(X, w=None, a=0, b=1, axis=0, kind_moment="s"):
    r""" Substitutes the rolling mean and divid by the rolling standard dev.

    .. math::

        RollStandardize(X)^w_t = b \times \frac{X_t - RollMean(X)^w_t}
        {RollStd(X)^w_t} + a

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Data to scale.
    w : int, optional
        Size of the lagged window of the moving average/standard deviation,
        must be positive. If ``w is None`` or ``w=0``, then
        ``w=X.shape[axis]``. Default is None.
    a, b : float or array_like, optional
        Respectively an additional and multiply factor.
    axis : int, optional
        Axis along which to scale the data.
    kind_moment : str {"s", "w", "e"}, optional
        - If "s" (default) then compute basic moving averages and standard
          deviations, see :func:`~fynance.features.momentums.sma` and
          :func:`~fynance.features.momentums.smstd`.
        - If "w" then compute the weighted moving averages and standard
          deviations, see :func:`~fynance.features.momentums.wma` and
          :func:`~fynance.features.momentums.wmstd`.
        - If "e" then compute the exponential moving averages and standard
          deviations, see :func:`~fynance.features.momentums.ema` and
          :func:`~fynance.features.momentums.emstd`.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        The scaled data.

    See Also
    --------
    Scale, normalize, standardize, roll_standardize

    """
    mean, std = _HANDLER_MOMENTUM[kind_moment]
    m = mean(X, w, axis=axis)
    s = std(X, w, axis=axis)

    if axis == 1:

        return _standardize(X.T, m, s, a, b).T

    return _standardize(X, m, s, a, b)


def normalize(X, a=0, b=1, axis=0):
    r""" Scale the data between ``a`` and ``b``.

    Substitutes the minimum and divid by the difference between the maximum and
    the minimum. Then multiply by ``b`` minus ``a`` and add ``a``.

    .. math::

        Normalize(X) = (b - a) \times \frac{X - X_{min}}{X_{max} - X_{min}} + a

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Data to scale.
    a, b : float or array_like, optional
        Respectively the lower and upper bound of the transformation.
    axis : int, optional
        Axis along which to scale the data.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        The scaled data.

    See Also
    --------
    Scale, standardize, roll_normalize

    """
    m = np.min(X, axis=axis)
    s = np.max(X, axis=axis)

    if axis == 1:

        return _normalize(X.T, m, s, a, b).T

    return _normalize(X, m, s, a, b)


def roll_normalize(X, w=None, a=0, b=1, axis=0):
    r""" Scale the data between ``a`` and ``b``.

    Substitutes the rolling minimum and divid by the difference between the
    rolling maximum and the minimum. Then multiply by ``b`` minus ``a`` and
    add ``a``.

    .. math::

        RollNormalize(X)^w_t = (b - a) \times \frac{X_t - RollMin(X)^w_t}
        {RollMax(X)^w_t - RollMin(X)^w_t} + a

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Data to scale.
    w : int, optional
        Size of the lagged window of the rolling minimum/maximum, must be
        positive. If ``w is None`` or ``w=0``, then ``w=X.shape[axis]``.
        Default is None.
    a, b : float or array_like, optional
        Respectively the lower and upper bound of the transformation.
    axis : int, optional
        Axis along which to scale the data.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        The scaled data.

    See Also
    --------
    Scale, standardize, normalize, roll_standardize

    """
    m = roll_min(X, w, axis=axis)
    s = roll_max(X, w, axis=axis)

    if axis == 1:

        return _normalize(X.T, m, s, a, b).T

    return _normalize(X, m, s, a, b)


if __name__ == "__main__":
    pass
