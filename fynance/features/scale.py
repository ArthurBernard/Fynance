#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2020-09-11 18:47:27
# @Last modified by: ArthurBernard
# @Last modified time: 2021-03-12 22:16:21

""" Data scaling utilities.

Functions and a fit/transform-style :class:`Scale` object to standardize
or normalize one- and two-dimensional arrays before feeding them to a
machine-learning model.

Both global versions (whole-sample mean/std, min/max) and rolling
versions (computed on a lagged window) are provided. Rolling variants
are lookahead-safe and recommended for time-series ML pipelines where
using a global statistic would leak future information into training
windows.

Main entry points
-----------------
- :func:`standardize` / :func:`roll_standardize` — z-score scaling.
- :func:`normalize` / :func:`roll_normalize` — min-max scaling.
- :class:`Scale` — fit/transform wrapper that stores parameters and
  exposes a :meth:`Scale.revert` inverse.

"""

# Built-in packages

# Third party packages
import numpy as np

from fynance.features.momentums import *

# Local packages
from fynance.features.roll_functions import roll_max, roll_min

__all__ = ["normalize", "roll_normalize", "roll_rank", "roll_standardize",
           "Scale", "standardize"]


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


def _get_roll_norm_params(X, w=None, axis=0):
    m = roll_min(X, w, axis=axis)
    s = roll_max(X, w, axis=axis)
    idx = m == s
    m[idx] = 0.
    s[idx] = 2 * s[idx]

    return {"m": m, "s": s}


def _get_roll_std_params(X, w=None, kind_moment="s", axis=0):
    m = _HANDLER_MOMENTUM[kind_moment][0](X, w=w, axis=axis)
    s = _HANDLER_MOMENTUM[kind_moment][1](X, w=w, axis=axis)
    s[s == 0] = 1.

    return {"m": m, "s": s}


class Scale:
    """ Fit/transform-style scaler for time-series data.

    Wraps the four scaling primitives (``standardize``, ``normalize``,
    ``roll_standardize``, ``roll_normalize``) behind a uniform fit /
    scale / revert API. Parameters are fitted once at construction
    and reused on subsequent calls — the typical pipeline pattern of
    fitting on a training window and applying the same transform to
    the test window, which avoids leaking test-period statistics into
    training.

    The ``revert`` method inverts the transformation, useful when the
    target of an ML model was scaled and the prediction must be
    converted back to the original units.

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
    handle_params: dict = {
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

    def __init__(self, X, kind="std", a=0., b=1., axis=0, **kwargs):
        """ Initialize the scale object. """
        self.func = self.handle_func[kind]
        self.revert_func = self.handle_revert[kind]
        self.kind = kind
        self.axis = axis
        self.fit(X, kind, a, b, axis, **kwargs)

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
                "".format(self.kind, self.params))

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

    Z-score scaling: shifts the data to zero mean and unit variance,
    then re-scales to ``[a, a + b]`` if the optional location/scale
    factors are provided. The standard preprocessing for ML models
    that assume features on comparable scales (linear regressions,
    SVMs, neural networks). For time-series with regime shifts, prefer
    :func:`roll_standardize` to avoid leaking future statistics.

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


def _roll_rank(X, w):
    X = np.asarray(X, dtype=np.float64)
    is1d = X.ndim == 1
    if is1d:
        X = X.reshape(-1, 1)
    out = np.full(X.shape, 0.5)
    for t in range(X.shape[0]):
        win = X[max(0, t - w + 1):t + 1]
        n = win.shape[0]
        if n > 1:
            out[t] = (win < X[t]).sum(axis=0) / (n - 1)

    return out[:, 0] if is1d else out


def roll_rank(X, w=None, axis=0):
    r""" Rolling percentile rank of each observation within its past window.

    For each ``t`` returns the fraction of the strictly-past window
    ``[t-w+1, t]`` that is below ``X_t``, in ``[0, 1]`` (0 = lowest, 1 =
    highest). Robust to outliers and strictly causal — useful as a feature
    normalisation that ignores the scale/distribution of the raw series.

    Parameters
    ----------
    X : np.ndarray[dtype, ndim=1 or 2]
        Data to rank.
    w : int, optional
        Size of the lagged window. If ``None`` or ``0``, ``w = X.shape[axis]``.
        Default None.
    axis : int, optional
        Axis along which to compute the rank. Default 0.

    Returns
    -------
    np.ndarray[dtype, ndim=1 or 2]
        Rolling percentile ranks in ``[0, 1]`` (0.5 for a single observation).

    See Also
    --------
    roll_normalize, roll_standardize

    Examples
    --------
    >>> X = np.array([1., 3., 2., 5., 4.])
    >>> roll_rank(X, w=3)
    array([0.5, 1. , 0.5, 1. , 0.5])

    """
    if w is None or w == 0:
        w = X.shape[axis]

    if axis == 1:

        return _roll_rank(X.T, w).T

    return _roll_rank(X, w)


if __name__ == "__main__":
    pass
