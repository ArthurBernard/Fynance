#!/usr/bin/env python3
# coding: utf-8

""" Tail-risk metrics — Value-at-Risk, Conditional VaR (Expected Shortfall),
Conditional Drawdown-at-Risk and a pairwise lower-tail dependence estimator.

Like :mod:`fynance.metrics.ratios`, ``var``/``cvar``/``cdar`` (and their rolling
variants) take a single equity/price curve and derive returns internally, so
they slot into the :data:`~fynance.metrics.summary.METRICS` registry alongside
``sharpe``/``sortino``/``calmar``. Unlike those ratios, the returns used here
are the *un-padded* per-period returns (see :func:`_returns_from_prices`): a
quantile or moment estimator must not see the placeholder zero return that
:func:`~fynance.features._metrics_helpers._compute_returns` prepends for the
unobserved period before ``X[0]``, or the tail estimate would be biased.
``tail_dependence`` is the one exception — it scores co-exceedance of two
series and so takes a ``(T, N)`` **returns panel** directly, mirroring
:func:`fynance.metrics.information_coefficient`'s (pred, real)-pair convention
rather than the single-curve one.

"""

from __future__ import annotations

# Built-in packages
from functools import lru_cache

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.stats import norm

# Local packages
from fynance.features._metrics_helpers import _drawdown

__all__ = ['var', 'cvar', 'cdar', 'roll_var', 'roll_cvar', 'tail_dependence']

_METHODS = ('historical', 'gaussian', 'cornish_fisher')
_CF_GRID_SIZE = 5000


def _check_alpha(alpha: float) -> None:
    if not (0. < alpha < 1.):

        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_method(method: str) -> None:
    if method not in _METHODS:

        raise ValueError(f"unknown method {method!r}, must be one of {_METHODS}")


def _returns_from_prices(X: NDArray) -> NDArray:
    """ Per-period returns of a 1-D price curve, ``R[i] = X[i+1] / X[i] - 1``.

    Deliberately **not** :func:`~fynance.features._metrics_helpers._compute_returns`
    (used by :func:`~fynance.metrics.sharpe` / :func:`~fynance.metrics.annual_volatility`):
    that helper prepends a placeholder ``R[0] = 0`` for the unobserved period
    before ``X[0]`` so the output keeps length ``T`` — harmless for a mean/std
    over a long series, but a fabricated zero return would bias a quantile or a
    skew/kurtosis estimate, especially on short samples. Tail-risk estimators
    use only the ``T - 1`` really observed returns.
    """
    X = np.asarray(X, dtype=np.float64)

    return X[1:] / X[:-1] - 1.


def _tail_k(alpha: float, n: int) -> int:
    """ Number of tail observations for confidence level ``alpha`` over ``n`` obs.

    ``k = max(1, floor(alpha * n))``: a deterministic count (no interpolation
    between two observations), so the ``k``-th smallest value is always an
    actually realized observation — the standard historical-simulation
    convention, and what keeps :func:`tail_dependence` symmetric (see there).
    """
    return max(1, int(np.floor(alpha * n)))


def _moments(R: NDArray) -> tuple[float, float, float, float]:
    """ Mean, population std, skewness and excess kurtosis of ``R``. """
    mu = float(np.mean(R))
    sigma = float(np.std(R))

    if sigma > 0.:
        z = (R - mu) / sigma
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.)
    else:
        skew = 0.
        kurt = 0.

    return mu, sigma, skew, kurt


def _cf_quantile(z, skew, kurt):
    """ Cornish-Fisher adjusted quantile (Favre & Galeano 2002), elementwise. """
    return (
        z + (z ** 2 - 1.) * skew / 6. + (z ** 3 - 3. * z) * kurt / 24.
        - (2. * z ** 3 - 5. * z) * skew ** 2 / 36.
    )


@lru_cache(maxsize=32)
def _cf_tail_constants(alpha: float, n_grid: int = _CF_GRID_SIZE) -> tuple[float, float, float, float]:
    """ Fixed coefficients (c1, c2, c3, c4) of the Cornish-Fisher tail mean.

    The Cornish-Fisher CVaR is defined as the tail mean of the CF-adjusted
    quantile function over the tail probability ``p in (0, alpha]``:

    .. math::

        CVaR_{CF} = -\\frac{1}{\\alpha}\\int_0^{\\alpha}
        \\left(\\mu + \\sigma \\, z_{CF}(p)\\right) dp

    Since ``z_CF(p) = z(p) + (z(p)^2-1) s/6 + (z(p)^3-3z(p)) k/24 -
    (2z(p)^3-5z(p)) s^2/36`` is *linear* in ``(s, k, s^2)`` for a fixed grid of
    ``z(p) = Phi^{-1}(p)``, the integral collapses to four constants — depending
    only on ``alpha`` and the grid, not on the sample — so both :func:`cvar` and
    :func:`roll_cvar` reduce to a cheap closed-form combination
    ``c1 + c2*skew + c3*kurt + c4*skew**2`` instead of a per-window numerical
    integration. Approximated by a fine midpoint grid on ``(0, alpha]``
    (``n_grid`` points); accurate to ~3e-4 relative error at the default size
    against the closed-form Gaussian limit (``skew = kurt = 0``).
    """
    p = np.linspace(alpha / n_grid, alpha, n_grid)
    z = norm.ppf(p)
    c1 = float(np.mean(z))
    c2 = float(np.mean(z ** 2 - 1.) / 6.)
    c3 = float(np.mean(z ** 3 - 3. * z) / 24.)
    c4 = float(-np.mean(2. * z ** 3 - 5. * z) / 36.)

    return c1, c2, c3, c4


# --------------------------------------------------------------------------- #
#   numba rolling kernels                                                     #
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _roll_window_moments_1d(R, w):
    """ Rolling mean/std/skew/excess-kurtosis of ``R`` over trailing window ``w``.

    NaN before the window is full (``t < w - 1``).
    """
    n = R.shape[0]
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    skew = np.full(n, np.nan)
    kurt = np.full(n, np.nan)

    for t in range(w - 1, n):
        m = 0.0
        for i in range(t - w + 1, t + 1):
            m += R[i]
        m /= w

        v = 0.0
        m3 = 0.0
        m4 = 0.0
        for i in range(t - w + 1, t + 1):
            d = R[i] - m
            v += d * d
            m3 += d ** 3
            m4 += d ** 4
        v /= w
        m3 /= w
        m4 /= w

        s = np.sqrt(v)
        mean[t] = m
        std[t] = s

        if s > 0.0:
            skew[t] = m3 / s ** 3
            kurt[t] = m4 / s ** 4 - 3.0
        else:
            skew[t] = 0.0
            kurt[t] = 0.0

    return mean, std, skew, kurt


@njit(cache=True)
def _roll_hist_quantile_1d(R, w, alpha):
    """ Rolling ``k``-th smallest return of ``R`` over trailing window ``w``. """
    n = R.shape[0]
    out = np.full(n, np.nan)
    k = max(1, int(np.floor(alpha * w)))

    for t in range(w - 1, n):
        window = np.sort(R[t - w + 1: t + 1])
        out[t] = window[k - 1]

    return out


@njit(cache=True)
def _roll_hist_tail_mean_1d(R, w, alpha):
    """ Rolling mean of the ``k`` smallest returns of ``R`` over window ``w``. """
    n = R.shape[0]
    out = np.full(n, np.nan)
    k = max(1, int(np.floor(alpha * w)))

    for t in range(w - 1, n):
        window = np.sort(R[t - w + 1: t + 1])
        s = 0.0
        for i in range(k):
            s += window[i]
        out[t] = s / k

    return out


# --------------------------------------------------------------------------- #
#   public API                                                                #
# --------------------------------------------------------------------------- #


def var(X: NDArray, alpha: float = 0.05, method: str = 'historical') -> float:
    r""" Value-at-Risk of a price/equity curve's per-period returns.

    The loss quantile at confidence level ``1 - alpha``: with probability
    ``alpha``, a period's return falls below ``-VaR``. Reported as a **positive**
    number (the magnitude of the loss quantile), following the usual risk-desk
    convention.

    Notes
    -----
    Let :math:`R` be the per-period returns of ``X`` (see
    :func:`_returns_from_prices`) and :math:`n` their count.

    - ``method='historical'``: the empirical quantile, taken as the ``k``-th
      smallest observed return (:func:`_tail_k`, ``k = max(1, floor(alpha n))``)
      — an actually realized scenario, no interpolation between two
      observations.
    - ``method='gaussian'``: :math:`VaR = -(\mu + \sigma z_\alpha)` with
      :math:`z_\alpha = \Phi^{-1}(\alpha)` the standard normal quantile,
      :math:`\mu, \sigma` the sample mean/std of :math:`R`.
    - ``method='cornish_fisher'``: as ``'gaussian'`` but with :math:`z_\alpha`
      replaced by the Cornish-Fisher expansion [1]_ that corrects for sample
      skewness :math:`s` and excess kurtosis :math:`k`:

      .. math::

          z_{CF} = z + \frac{z^2 - 1}{6}s + \frac{z^3 - 3z}{24}k
          - \frac{2z^3 - 5z}{36}s^2

      Reduces to the Gaussian quantile when :math:`s = k = 0`.

    Parameters
    ----------
    X : array_like
        Time-series of price, performance or index (a single curve).
    alpha : float, optional
        Tail probability, in ``(0, 1)``. Default is 0.05 (95% VaR).
    method : {'historical', 'gaussian', 'cornish_fisher'}, optional
        Estimation method. Default is 'historical'.

    Returns
    -------
    float
        Value-at-Risk, positive for a typical loss-bearing distribution.

    References
    ----------
    .. [1] Favre, L., and Galeano, J.-A., 2002, Mean-Modified Value-at-Risk
       Optimization with Hedge Funds, Journal of Alternative Investments.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([100., 99., 103., 95., 101., 98., 104., 90., 108., 97.,
    ...                102.])
    >>> round(var(X, alpha=0.2, method='historical'), 4)
    0.1019
    >>> round(var(X, alpha=0.2, method='gaussian'), 4)
    0.0723

    See Also
    --------
    cvar, cdar, roll_var, tail_dependence

    """
    _check_alpha(alpha)
    _check_method(method)
    R = _returns_from_prices(X)

    if method == 'historical':
        k = _tail_k(alpha, R.shape[0])
        q = np.sort(R)[k - 1]
    elif method == 'gaussian':
        mu, sigma, _, _ = _moments(R)
        q = mu + sigma * norm.ppf(alpha)
    else:
        mu, sigma, skew, kurt = _moments(R)
        q = mu + sigma * _cf_quantile(norm.ppf(alpha), skew, kurt)

    return float(-q)


def cvar(X: NDArray, alpha: float = 0.05, method: str = 'historical') -> float:
    r""" Conditional Value-at-Risk (Expected Shortfall) of a price/equity curve.

    The expected loss *beyond* the VaR threshold — the mean of the returns in
    the ``alpha``-tail — a coherent risk measure (unlike VaR, which is not
    subadditive). Reported as a **positive** number, same convention as
    :func:`var`.

    Notes
    -----
    Let :math:`R` be the per-period returns of ``X`` (see
    :func:`_returns_from_prices`), :math:`n` their count and :math:`\mu,\sigma`
    their sample mean/std.

    - ``method='historical'``: mean of the ``k = max(1, floor(alpha n))``
      smallest observed returns (:func:`_tail_k`) — the tail mean underlying
      :func:`var`'s historical quantile.
    - ``method='gaussian'``: closed form [2]_
      :math:`CVaR = -\mu + \sigma \frac{\varphi(z_\alpha)}{\alpha}`, with
      :math:`\varphi` the standard normal pdf and :math:`z_\alpha =
      \Phi^{-1}(\alpha)`.
    - ``method='cornish_fisher'``: the tail mean of the Cornish-Fisher [1]_
      -adjusted quantile function over :math:`p \in (0, \alpha]`,

      .. math::

          CVaR_{CF} = -\frac{1}{\alpha}\int_0^{\alpha}
          \left(\mu + \sigma \, z_{CF}(p)\right) dp

      approximated on a fixed probability grid (see
      :func:`_cf_tail_constants`); collapses to the same value as
      ``'gaussian'`` when sample skewness and excess kurtosis are both zero.

    Parameters
    ----------
    X : array_like
        Time-series of price, performance or index (a single curve).
    alpha : float, optional
        Tail probability, in ``(0, 1)``. Default is 0.05.
    method : {'historical', 'gaussian', 'cornish_fisher'}, optional
        Estimation method. Default is 'historical'.

    Returns
    -------
    float
        Conditional Value-at-Risk, positive for a typical loss-bearing
        distribution. Always :math:`\geq` :func:`var` at the same ``alpha``.

    References
    ----------
    .. [1] Favre, L., and Galeano, J.-A., 2002, Mean-Modified Value-at-Risk
       Optimization with Hedge Funds, Journal of Alternative Investments.
    .. [2] Rockafellar, R.T., and Uryasev, S., 2000, Optimization of
       Conditional Value-at-Risk, Journal of Risk, 2, 21-42.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([100., 99., 103., 95., 101., 98., 104., 90., 108., 97.,
    ...                102.])
    >>> round(cvar(X, alpha=0.2, method='historical'), 4)
    0.1182
    >>> cvar(X, alpha=0.2, method='historical') >= var(X, alpha=0.2, method='historical')
    True

    See Also
    --------
    var, cdar, roll_cvar, tail_dependence

    """
    _check_alpha(alpha)
    _check_method(method)
    R = _returns_from_prices(X)

    if method == 'historical':
        k = _tail_k(alpha, R.shape[0])
        raw = float(np.mean(np.sort(R)[:k]))
    elif method == 'gaussian':
        mu, sigma, _, _ = _moments(R)
        z = norm.ppf(alpha)
        raw = mu - sigma * norm.pdf(z) / alpha
    else:
        mu, sigma, skew, kurt = _moments(R)
        c1, c2, c3, c4 = _cf_tail_constants(alpha)
        raw = mu + sigma * (c1 + c2 * skew + c3 * kurt + c4 * skew ** 2)

    return float(-raw)


def cdar(X: NDArray, alpha: float = 0.05) -> float:
    r""" Conditional Drawdown-at-Risk of a price/equity curve.

    Mean of the ``alpha`` worst drawdown depths [3]_ observed over the series —
    the drawdown analogue of :func:`cvar`: while :func:`~fynance.metrics.mdd`
    reports only the single worst drawdown, CDaR averages the whole worst tail,
    making it less sensitive to a single outlier path.

    Notes
    -----
    Let :math:`DD` be the percentage drawdown path of ``X`` (see
    :func:`~fynance.metrics.drawdown`) and :math:`T` its length:

    .. math::

        CDaR_\alpha = \frac{1}{k}\sum_{i=1}^{k} DD_{(T + 1 - i)}

    where :math:`DD_{(1)} \leq \dots \leq DD_{(T)}` is :math:`DD` sorted
    ascending and :math:`k = \max(1, \lfloor \alpha T \rfloor)`
    (:func:`_tail_k`) — the same tail-count convention as :func:`var`/:func:`cvar`.

    Parameters
    ----------
    X : array_like
        Time-series of price, performance or index (a single curve). Must be
        positive values (see :func:`~fynance.metrics.drawdown`).
    alpha : float, optional
        Tail fraction, in ``(0, 1)``. Default is 0.05.

    Returns
    -------
    float
        Conditional Drawdown-at-Risk, in ``[0, 1]``. Always :math:`\geq`
        :func:`~fynance.metrics.mdd` divided by 1 (it is a tail *mean*, so
        :math:`\leq` the maximum drawdown).

    References
    ----------
    .. [3] Chekhlov, A., Uryasev, S., and Zabarankin, M., 2005, Drawdown
       Measure in Portfolio Optimization, International Journal of Theoretical
       and Applied Finance, 8(1), 13-58.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([100., 90., 95., 80., 110., 70., 120., 60., 130.])
    >>> round(cdar(X, alpha=0.3), 4)
    0.4318

    See Also
    --------
    cvar, var, mdd, drawdown

    """
    _check_alpha(alpha)
    X = np.asarray(X, dtype=np.float64)
    dd = _drawdown(X, False)
    k = _tail_k(alpha, dd.shape[0])
    worst = np.sort(dd)[::-1][:k]

    return float(np.mean(worst))


def roll_var(X: NDArray, alpha: float = 0.05, w: int = 252, method: str = 'historical') -> NDArray:
    r""" Rolling Value-at-Risk of a price/equity curve over a trailing window.

    Causal trailing estimate of :func:`var`: each output point uses only the
    ``w`` returns observed up to and including that point, so the output has a
    ``w``-point NaN head (there is no well-defined estimate before the window
    fills).

    Parameters
    ----------
    X : array_like
        Time-series of price, performance or index (a single curve), shape
        ``(T,)``.
    alpha : float, optional
        Tail probability, in ``(0, 1)``. Default is 0.05.
    w : int, optional
        Size of the trailing window, in number of returns. Default is 252.
    method : {'historical', 'gaussian', 'cornish_fisher'}, optional
        Estimation method, see :func:`var`. Default is 'historical'.

    Returns
    -------
    np.ndarray[np.float64, ndim=1] of shape (T,)
        Rolling Value-at-Risk; the first ``w`` values are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([100., 99., 103., 95., 101., 98., 104., 90., 108., 97.,
    ...                102.])
    >>> out = roll_var(X, alpha=0.4, w=5, method='historical')
    >>> np.isnan(out[:5]).all()
    True
    >>> round(out[5], 4)
    0.0297

    See Also
    --------
    var, roll_cvar, roll_mdd

    """
    _check_alpha(alpha)
    _check_method(method)
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]
    R = _returns_from_prices(X)

    if method == 'historical':
        q = _roll_hist_quantile_1d(R, w, alpha)
    elif method == 'gaussian':
        mean, std, _, _ = _roll_window_moments_1d(R, w)
        q = mean + std * norm.ppf(alpha)
    else:
        mean, std, skew, kurt = _roll_window_moments_1d(R, w)
        q = mean + std * _cf_quantile(norm.ppf(alpha), skew, kurt)

    out = np.full(T, np.nan)
    out[1:] = -q

    return out


def roll_cvar(X: NDArray, alpha: float = 0.05, w: int = 252, method: str = 'historical') -> NDArray:
    r""" Rolling Conditional Value-at-Risk of a price/equity curve.

    Causal trailing estimate of :func:`cvar`: each output point uses only the
    ``w`` returns observed up to and including that point, so the output has a
    ``w``-point NaN head.

    Parameters
    ----------
    X : array_like
        Time-series of price, performance or index (a single curve), shape
        ``(T,)``.
    alpha : float, optional
        Tail probability, in ``(0, 1)``. Default is 0.05.
    w : int, optional
        Size of the trailing window, in number of returns. Default is 252.
    method : {'historical', 'gaussian', 'cornish_fisher'}, optional
        Estimation method, see :func:`cvar`. Default is 'historical'.

    Returns
    -------
    np.ndarray[np.float64, ndim=1] of shape (T,)
        Rolling Conditional Value-at-Risk; the first ``w`` values are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([100., 99., 103., 95., 101., 98., 104., 90., 108., 97.,
    ...                102.])
    >>> out = roll_cvar(X, alpha=0.4, w=5, method='historical')
    >>> np.isnan(out[:5]).all()
    True
    >>> round(out[5], 4)
    0.0537

    See Also
    --------
    cvar, roll_var, roll_mdd

    """
    _check_alpha(alpha)
    _check_method(method)
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]
    R = _returns_from_prices(X)

    if method == 'historical':
        raw = _roll_hist_tail_mean_1d(R, w, alpha)
    elif method == 'gaussian':
        mean, std, _, _ = _roll_window_moments_1d(R, w)
        z = norm.ppf(alpha)
        raw = mean - std * norm.pdf(z) / alpha
    else:
        mean, std, skew, kurt = _roll_window_moments_1d(R, w)
        c1, c2, c3, c4 = _cf_tail_constants(alpha)
        raw = mean + std * (c1 + c2 * skew + c3 * kurt + c4 * skew ** 2)

    out = np.full(T, np.nan)
    out[1:] = -raw

    return out


def tail_dependence(X: NDArray, q: float = 0.05) -> NDArray:
    r""" Pairwise lower-tail dependence of a ``(T, N)`` returns panel.

    Unlike :func:`var`/:func:`cvar`/:func:`cdar`, this takes returns directly
    (see the module docstring): tail dependence is a *co*-exceedance property of
    two series, not a summary of one equity curve, so it does not fit the
    single-curve ``METRICS`` contract (mirroring
    :func:`~fynance.metrics.information_coefficient`'s (pred, real)-pair
    convention).

    Estimates, for each pair of assets :math:`(i, j)`, the empirical
    probability [4]_ that asset :math:`i` is in its own lower ``q``-tail given
    that asset :math:`j` is in its own lower ``q``-tail:

    .. math::

        \lambda_{ij} = P\left(x_i \leq Q_i(q) \mid x_j \leq Q_j(q)\right)

    Notes
    -----
    :math:`Q_i(q)` is the empirical ``q``-quantile of column ``i`` (the same
    ``k``-th-smallest-observation convention as :func:`var`, :func:`_tail_k`),
    so by construction exactly :math:`k = \max(1, \lfloor qT \rfloor)`
    observations of every column are ``<= `` its own threshold. The estimator
    is then

    .. math::

        \lambda_{ij} = \frac{\#\{t : x_{ti} \leq Q_i(q) \text{ and } x_{tj}
        \leq Q_j(q)\}}{k}

    Because the denominator ``k`` is the same for both conditioning directions
    (it only depends on ``q`` and ``T``, not on which column conditions on
    which), :math:`\lambda_{ij} = \lambda_{ji}`: the matrix is symmetric by
    construction, unlike a generic conditional probability. The diagonal is
    exactly 1 (an event's exceedance is always joint with itself).

    Parameters
    ----------
    X : array_like of shape (T, N)
        Returns panel (not prices), one column per asset.
    q : float, optional
        Tail probability, in ``(0, 1)``. Default is 0.05.

    Returns
    -------
    np.ndarray[np.float64, ndim=2] of shape (N, N)
        Symmetric lower-tail dependence matrix, diagonal 1.

    References
    ----------
    .. [4] Longin, F., and Solnik, B., 2001, Extreme Correlation of
       International Equity Markets, Journal of Finance, 56(2), 649-676.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> comonotone = rng.standard_normal(2000)
    >>> R = np.stack([comonotone, comonotone], axis=1)
    >>> lam = tail_dependence(R, q=0.05)
    >>> bool(lam[0, 1] > 0.95)
    True
    >>> np.diag(lam)
    array([1., 1.])

    See Also
    --------
    var, information_coefficient

    """
    _check_alpha(q)
    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    k = _tail_k(q, T)

    thresholds = np.sort(X, axis=0)[k - 1, :]
    exceed = (X <= thresholds).astype(np.float64)

    out = (exceed.T @ exceed) / k
    np.fill_diagonal(out, 1.0)

    return out
