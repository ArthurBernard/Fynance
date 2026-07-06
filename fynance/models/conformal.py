#!/usr/bin/env python3
# coding: utf-8

""" Causal split-conformal prediction intervals.

Wraps any :class:`~fynance.core.protocols.SignalModel`-conforming regressor
(``fit(X, y)`` / ``predict(X)``) with a distribution-free prediction
interval, calibrated on a **trailing, strictly past** slice of the data so
the calibration step never leaks future information into the interval it
reports for past bars.

**Split-conformal recap.** Given a fitted point-prediction model and a
held-out calibration set of ``n`` residuals :math:`|y_i - \\hat y_i|`, the
classic split-conformal method [1]_ [2]_ takes

.. math::

    \\hat q = \\text{the } \\lceil (n+1)(1-\\alpha) \\rceil \\text{-th smallest of
    the } n \\text{ calibration residuals}

and reports :math:`\\hat y(x) \\pm \\hat q` as a marginally valid
:math:`(1-\\alpha)` prediction interval for a fresh, exchangeable
observation. When :math:`\\lceil (n+1)(1-\\alpha) \\rceil` exceeds ``n`` (which
happens for small ``n`` and/or small ``alpha``, e.g. ``n=10, alpha=0.05``)
the theoretical interval is unbounded; this implementation instead caps the
index at ``n`` (i.e. uses the largest calibration residual), the common
practical convention -- documented here rather than silently clamped.

Because :math:`\\hat q` is a single scalar fit once at calibration time, the
resulting interval has **constant width**: it is marginally calibrated (the
long-run average coverage is :math:`1-\\alpha`) even under heteroskedastic
data, but it is not *conditionally* calibrated -- it will under-cover in
high-volatility regimes and over-cover in low-volatility ones. See
:func:`rolling_conformal` for a walk-forward wrapper that at least lets
``q_hat`` adapt across calibration windows over time.

Main entry points
------------------
- :class:`ConformalWrapper` -- fit/calibrate/predict a single split.
- :func:`rolling_conformal` -- walk-forward train/calibrate/test loop.

References
----------
.. [1] Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in
   a Random World*. Springer.
.. [2] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L.
   (2018). Distribution-Free Predictive Inference for Regression. *Journal
   of the American Statistical Association*, 113(523), 1094-1111.

"""

from __future__ import annotations

# Built-in packages
from typing import Any, Callable

# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = ['ConformalWrapper', 'rolling_conformal']


def _split_conformal_quantile(residuals: NDArray, alpha: float) -> float:
    """ Compute the split-conformal quantile of calibration residuals.

    Parameters
    ----------
    residuals : numpy.ndarray
        Non-negative calibration residuals, shape ``(n,)``, ``n >= 1``.
    alpha : float
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    float
        The :math:`\\lceil (n+1)(1-\\alpha) \\rceil`-th smallest residual
        (1-indexed order statistic, capped at ``n``).

    """
    n = residuals.shape[0]
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)

    return float(np.sort(residuals)[k - 1])


class ConformalWrapper:
    """ Causal split-conformal prediction interval around a point model.

    Wraps a ``model`` exposing ``fit(X, y)`` / ``predict(X)`` (the
    :class:`~fynance.core.protocols.SignalModel` contract). :meth:`fit`
    trains ``model`` on the leading ``T - window`` bars only and calibrates
    the interval half-width ``q_hat`` on the absolute residuals of the
    trailing ``window`` bars -- bars the wrapped model never saw during its
    own fit, so the calibration is honest (no in-sample residuals).

    Parameters
    ----------
    model : SignalModel
        Any regressor exposing ``fit(X, y)`` (returns anything, ``self`` is
        conventional) and ``predict(X) -> array-like``.
    alpha : float, optional
        Miscoverage level in ``(0, 1)``; the interval targets marginal
        coverage ``1 - alpha``. Default 0.1 (90% interval).
    window : int, optional
        Number of trailing bars held out for calibration. Must be strictly
        less than the number of observations passed to :meth:`fit`. Default
        252 (one trading year of daily bars).

    Attributes
    ----------
    q_hat_ : float or None
        Calibrated half-width, set by :meth:`fit`; ``None`` before fitting.

    Examples
    --------
    A closed-form linear model (OLS), fit/calibrated on noiseless-ish
    synthetic data -- the interval should be tight and cover the held-out
    calibration residuals by construction:

    >>> import numpy as np
    >>> from fynance.models.conformal import ConformalWrapper
    >>> class LinearModel:
    ...     ''' Closed-form ordinary least squares. '''
    ...     def fit(self, X, y):
    ...         self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
    ...         return self
    ...     def predict(self, X):
    ...         return X @ self.coef_
    >>> rng = np.random.default_rng(0)
    >>> T, F = 500, 2
    >>> X = rng.standard_normal((T, F))
    >>> w_true = np.array([1.5, -0.5])
    >>> y = X @ w_true + 0.1 * rng.standard_normal(T)
    >>> wrapper = ConformalWrapper(LinearModel(), alpha=0.1, window=100)
    >>> _ = wrapper.fit(X, y)
    >>> wrapper.q_hat_ > 0
    True
    >>> interval = wrapper.predict_interval(X[-5:])
    >>> interval.shape
    (5, 2)
    >>> bool((interval[:, 1] > interval[:, 0]).all())
    True

    See Also
    --------
    rolling_conformal, fynance.core.protocols.SignalModel

    """

    def __init__(self, model: Any, alpha: float = 0.1, window: int = 252) -> None:
        """ Initialize the wrapper (see class docstring for parameters). """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.model = model
        self.alpha = alpha
        self.window = window
        self.q_hat_: float | None = None

    def fit(self, X: NDArray, y: NDArray) -> ConformalWrapper:
        """ Fit the wrapped model and calibrate the interval half-width.

        Splits ``(X, y)`` at ``T - window``: the wrapped model is fit on the
        leading slice only; the trailing ``window`` bars are held out and
        never used for fitting, only to compute calibration residuals
        ``|y - model.predict(X)|`` and their split-conformal quantile
        (:data:`q_hat_`).

        Parameters
        ----------
        X : array-like
            Feature matrix, shape ``(T, N)``, time-ordered.
        y : array-like
            Target, shape ``(T,)`` or ``(T, 1)``, aligned with ``X``.

        Returns
        -------
        ConformalWrapper
            ``self``, to allow chaining (the
            :class:`~fynance.core.protocols.SignalModel` contract).

        Raises
        ------
        ValueError
            If ``window`` is not strictly less than the number of
            observations in ``X``/``y`` (there would be no leading slice
            left to fit the wrapped model on).

        """
        X = np.asarray(X)
        y_arr = np.asarray(y)
        T = y_arr.shape[0]

        if not self.window < T:
            raise ValueError(
                f"window ({self.window}) must be < the number of "
                f"observations ({T}); nothing would be left to fit on"
            )

        split = T - self.window
        # Pass y through with its original shape -- e.g. a torch-style
        # wrapped model may require (T, M), not flattened (T,).
        self.model.fit(X[:split], y_arr[:split])

        cal_pred = np.asarray(self.model.predict(X[split:])).reshape(-1)
        y_cal = y_arr[split:].reshape(-1)
        residuals = np.abs(y_cal - cal_pred)
        self.q_hat_ = _split_conformal_quantile(residuals, self.alpha)

        return self

    def predict(self, X: NDArray) -> NDArray:
        """ Point predictions of the wrapped model.

        Parameters
        ----------
        X : array-like
            Feature matrix, shape ``(T, N)``.

        Returns
        -------
        numpy.ndarray
            Point predictions, shape ``(T,)`` (the
            :class:`~fynance.core.protocols.SignalModel` contract).

        """
        return np.asarray(self.model.predict(X)).reshape(-1)

    def predict_interval(self, X: NDArray) -> NDArray:
        """ Constant-width prediction interval around the point prediction.

        Parameters
        ----------
        X : array-like
            Feature matrix, shape ``(T, N)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(T, 2)``; column 0 is the lower bound
            (``predict(X) - q_hat_``), column 1 the upper bound
            (``predict(X) + q_hat_``). The width ``2 * q_hat_`` is the same
            for every row -- see the module docstring's note on the
            constant-width limitation under heteroskedasticity.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        """
        if self.q_hat_ is None:
            raise RuntimeError("ConformalWrapper.predict_interval called before fit")

        point = self.predict(X)

        return np.stack([point - self.q_hat_, point + self.q_hat_], axis=1)


def rolling_conformal(
    model_factory: Callable[[], Any],
    X: NDArray,
    y: NDArray,
    *,
    train: int = 252,
    cal: int = 63,
    test: int = 63,
    alpha: float = 0.1,
) -> dict[str, NDArray | float]:
    """ Walk-forward split-conformal prediction intervals.

    Rolls a strictly causal train / calibrate / test window over
    ``(X, y)``: at each step, a **fresh** model (from ``model_factory()``) is
    fit on ``train`` bars, calibrated (à la :class:`ConformalWrapper`) on the
    next ``cal`` bars, and used to emit point predictions and intervals on
    the next ``test`` bars; the window then advances by ``test`` bars (the
    step size), so consecutive test windows never overlap. Every prediction
    and interval reported for bar ``t`` depends only on bars strictly before
    the test window containing ``t`` -- perturbing ``X``/``y`` at or after a
    test window leaves every earlier output bit-for-bit unchanged.

    :func:`fynance.data.split.walk_forward` is not reused here because it
    only generates a 2-way (train/test) split; the 3-way train/calibrate/test
    arithmetic needed for conformal calibration is implemented directly
    below.

    Parameters
    ----------
    model_factory : callable
        No-argument callable returning a **fresh** model exposing
        ``fit(X, y)`` / ``predict(X)`` (the
        :class:`~fynance.core.protocols.SignalModel` contract) each time it
        is called -- one fresh model per window, so no state (and no
        information) leaks across windows.
    X : array-like
        Feature matrix, shape ``(T, N)``, time-ordered.
    y : array-like
        Target, shape ``(T,)`` or ``(T, 1)``, aligned with ``X``.
    train, cal, test : int, optional
        Train, calibration and test window lengths in bars. Defaults
        252/63/63 (roughly one trading year of training, one quarter of
        calibration, one quarter of test).
    alpha : float, optional
        Miscoverage level in ``(0, 1)``; each window's interval targets
        marginal coverage ``1 - alpha``. Default 0.1.

    Returns
    -------
    dict
        - ``'pred'``, ``'lo'``, ``'hi'`` : numpy.ndarray, shape ``(T,)`` --
          point prediction and interval bounds, ``NaN`` outside any test
          window (i.e. before the first window's test slice, and in any
          trailing bars too short to form one more full window).
        - ``'covered'`` : numpy.ndarray, shape ``(T,)`` -- ``1.0`` where
          ``lo <= y <= hi``, ``0.0`` where not covered, ``NaN`` outside any
          test window (NaN-aware boolean mask).
        - ``'coverage'`` : float -- ``nanmean`` of ``'covered'``, the
          realized marginal coverage over all test bars; ``NaN`` if no test
          window was ever produced (``train + cal + test > T``).

    Raises
    ------
    ValueError
        If ``train``, ``cal`` or ``test`` is not strictly positive, or if
        ``alpha`` is not in ``(0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.models.conformal import rolling_conformal
    >>> class LinearModel:
    ...     ''' Closed-form ordinary least squares. '''
    ...     def fit(self, X, y):
    ...         self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
    ...         return self
    ...     def predict(self, X):
    ...         return X @ self.coef_
    >>> rng = np.random.default_rng(0)
    >>> T, F = 600, 2
    >>> X = rng.standard_normal((T, F))
    >>> w_true = np.array([1.0, -1.0])
    >>> y = X @ w_true + 0.1 * rng.standard_normal(T)
    >>> result = rolling_conformal(
    ...     LinearModel, X, y, train=200, cal=50, test=50, alpha=0.1,
    ... )
    >>> sorted(result.keys())
    ['coverage', 'covered', 'hi', 'lo', 'pred']
    >>> result['pred'].shape
    (600,)
    >>> 0.7 < result['coverage'] <= 1.0
    True

    See Also
    --------
    ConformalWrapper, fynance.data.split.walk_forward

    """
    if train <= 0 or cal <= 0 or test <= 0:
        raise ValueError(
            f"train, cal and test must be > 0, got train={train}, cal={cal}, "
            f"test={test}"
        )

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]

    pred = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    covered = np.full(n, np.nan)

    window = train + cal
    t = 0

    while t + window + test <= n:
        wrapper = ConformalWrapper(model_factory(), alpha=alpha, window=cal)
        wrapper.fit(X[t:t + window], y[t:t + window])

        test_slice = slice(t + window, t + window + test)
        X_test, y_test = X[test_slice], y[test_slice]

        interval = wrapper.predict_interval(X_test)
        pred[test_slice] = wrapper.predict(X_test)
        lo[test_slice] = interval[:, 0]
        hi[test_slice] = interval[:, 1]
        covered[test_slice] = (
            (y_test >= interval[:, 0]) & (y_test <= interval[:, 1])
        ).astype(np.float64)

        t += test

    with np.errstate(invalid='ignore'):
        coverage = float(np.nanmean(covered)) if np.any(~np.isnan(covered)) else float('nan')

    return {'pred': pred, 'lo': lo, 'hi': hi, 'covered': covered, 'coverage': coverage}
