#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-11 18:47:27
# @Last modified by: ArthurBernard
# @Last modified time: 2020-09-11 21:10:51

""" Object to scale data. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages

__all__ = ["normalize", "Scale", "standardize"]


# TODO :
#     - Use wrapper for axis for scale methods
#     - Use wrapper for axis for standardize and normalize functions
#     - Append functions or method to scale with moving functions.


def _get_norm_params(X, axis=0):
    params = {
        "m": np.min(X, axis=axis),
        "s": np.max(X, axis=axis),
    }

    return params


def _get_std_params(X, axis=0):
    params = {
        "m": np.mean(X, axis=axis),
        "s": np.std(X, axis=axis),
    }

    return params


def _normalize(X, m, s, a, b):

    return (b - a) * (X - m) / (s - m) + a


def _standardize(X, m, s, a, b):

    return b * (X - m) / s + a


def _revert_normalize(X, m, s, a, b):

    return _normalize(X, a, b, m, s)


def _revert_standardize(X, m, s, a, b):

    return _standardize(X, a, b, m, s)


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
    a, b : float or array_like, optional
        Some scale factors to apply after the transformation. By default is
        respectively 0 and 1.
    axis : int, optional
        Axis along which compute the scale parameters. Default is 0.

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
    normalize, standardize

    """

    handle_func = {
        "raw": lambda x, a, b: x,
        "std": _standardize,
        "norm": _normalize,
    }
    handle_params = {
        "raw": lambda x: {},
        "std": _get_std_params,
        "norm": _get_norm_params,
    }
    handle_revert = {
        "raw": lambda x, a, b: x,
        "std": _revert_standardize,
        "norm": _revert_normalize,
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

    def fit(self, X, kind=None, a=0., b=1., axis=0):
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
        a, b : float or array_like, optional
            Some scale factors to apply after the transformation. By default is
            respectively 0 and 1.
        axis : int, optional
            Axis along which compute the scale parameters. Default is 0.

        """
        if kind is None:
            kind = self.kind

        if axis is None:
            axis = self.axis

        self.params = self.handle_params[kind](X, axis=axis)
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
    Scale, normalize

    """
    m = np.mean(X, axis=axis)
    s = np.std(X, axis=axis)

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
    Scale, standardize

    """
    m = np.min(X, axis=axis)
    s = np.max(X, axis=axis)

    if axis == 1:

        return _normalize(X.T, m, s, a, b).T

    return _normalize(X, m, s, a, b)


if __name__ == "__main__":
    pass
