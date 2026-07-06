#!/usr/bin/env python3
# coding: utf-8

""" Executable house-rule checks: protocol conformance and causality.

Turns two standing rules of the fynance pipeline into runnable assertions that
downstream ``pytest`` suites can import directly:

- **Protocol conformance** (:func:`check_conforms`) smoke-runs the seams
  defined in :mod:`fynance.core.protocols` (:class:`~fynance.core.protocols.
  FeatureTransform`, :class:`~fynance.core.protocols.SignalModel`, ...) on
  small seeded synthetic data, catching a wrong return type/shape/dtype before
  it reaches the rest of the pipeline.
- **No lookahead** (:func:`assert_causal`) turns the "no lookahead bias"
  house rule (see the repository invariants and :mod:`fynance.core.
  protocols`) into a black-box probe: perturb the input strictly after an
  index ``t0`` and require the output strictly before ``t0`` to be
  unchanged.

Both raise :class:`AssertionError` with an actionable message (naming the
offending method or the earliest leaking index) rather than a bare
``assert``, so failures are diagnosable straight from the pytest output.

"""

from __future__ import annotations

# Built-in packages
from typing import Any, Callable, Iterator

# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages
from fynance.core.protocols import (
    Allocator,
    CostModel,
    DataSource,
    FeatureTransform,
    Metric,
    SignalModel,
)

__all__ = ['check_conforms', 'assert_causal']


# =========================================================================== #
#                        Protocol conformance smoke test                      #
# =========================================================================== #


def _expect_ndarray(
    value: Any, method: str, expected_shape: tuple[int, ...] | None = None,
) -> None:
    """ Raise ``AssertionError`` unless `value` is an ndarray of the expected shape. """
    if not isinstance(value, np.ndarray):
        suffix = f" of shape {expected_shape}" if expected_shape is not None else ""
        raise AssertionError(
            f"{method}() returned {type(value).__name__}, expected "
            f"numpy.ndarray{suffix}"
        )

    if expected_shape is not None and value.shape != expected_shape:
        raise AssertionError(
            f"{method}() returned numpy.ndarray of shape {value.shape}, "
            f"expected shape {expected_shape}"
        )


def _expect_scalar(value: Any, method: str) -> None:
    """ Raise ``AssertionError`` unless `value` is a plain scalar number. """
    if isinstance(value, np.ndarray) or not isinstance(
        value, (int, float, np.floating, np.integer),
    ):
        raise AssertionError(
            f"{method}() returned {type(value).__name__}, expected float"
        )


def _call(method: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """ Call `fn` and re-raise any failure as an actionable ``AssertionError``. """
    try:
        return fn(*args, **kwargs)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"{method}() raised {type(exc).__name__}: {exc}") from exc


def _smoke_feature_transform(obj: Any, rng: np.random.Generator, T: int, N: int) -> None:
    """ ``fit(X)`` then ``transform(X)`` on an ``(T, N)`` float64 array. """
    X = rng.standard_normal((T, N))
    _call('fit', obj.fit, X)
    out = _call('transform', obj.transform, X)
    _expect_ndarray(out, 'transform')
    if out.shape[0] != T:
        raise AssertionError(
            f"transform() returned numpy.ndarray of shape {out.shape}, "
            f"expected a leading (time) axis of length {T}"
        )


def _smoke_signal_model(obj: Any, rng: np.random.Generator, T: int, N: int) -> None:
    """ ``fit(X, y)`` then ``predict(X)`` on ``(T, N)`` features / ``(T,)`` target. """
    X = rng.standard_normal((T, N))
    y = rng.standard_normal((T,))
    _call('fit', obj.fit, X, y)
    out = _call('predict', obj.predict, X)
    _expect_ndarray(out, 'predict', expected_shape=(T,))


def _smoke_allocator(obj: Any, rng: np.random.Generator, T: int, N: int) -> None:
    """ ``__call__(data)`` on an ``(T, N)`` return/covariance-like matrix. """
    data = rng.standard_normal((T, N))
    out = _call('__call__', obj, data)
    _expect_ndarray(out, '__call__', expected_shape=(N,))


def _smoke_cost_model(obj: Any, rng: np.random.Generator, T: int, N: int) -> None:
    """ ``__call__(weights)`` on an ``(T, N)`` weight path. """
    weights = rng.uniform(0.0, 1.0, size=(T, N))
    out = _call('__call__', obj, weights)
    _expect_ndarray(out, '__call__', expected_shape=(T,))


def _smoke_metric(obj: Any, rng: np.random.Generator, T: int, N: int) -> None:
    """ ``__call__(returns)`` on a ``(T,)`` return curve. """
    returns = rng.standard_normal((T,))
    out = _call('__call__', obj, returns)
    _expect_scalar(out, '__call__')


_RECIPES: dict[type, Callable[[Any, np.random.Generator, int, int], None]] = {
    FeatureTransform: _smoke_feature_transform,
    SignalModel: _smoke_signal_model,
    Allocator: _smoke_allocator,
    CostModel: _smoke_cost_model,
    Metric: _smoke_metric,
}


def check_conforms(
    obj: Any, protocol: type, *, T: int = 64, N: int = 3, seed: int = 0,
) -> None:
    r""" Smoke-run `protocol`'s methods on `obj` with seeded synthetic data.

    Designs one canonical smoke input per protocol from the actual method
    set in :mod:`fynance.core.protocols`:

    - :class:`~fynance.core.protocols.FeatureTransform`: ``fit(X)`` then
      ``transform(X)`` on a ``(T, N)`` array; ``transform`` must return a
      ``numpy.ndarray`` whose leading (time) axis has length ``T``.
    - :class:`~fynance.core.protocols.SignalModel`: ``fit(X, y)`` then
      ``predict(X)``; ``predict`` must return a ``numpy.ndarray`` of shape
      ``(T,)``.
    - :class:`~fynance.core.protocols.Allocator`: ``__call__`` on a
      ``(T, N)`` return/covariance-like matrix; must return a
      ``numpy.ndarray`` of shape ``(N,)`` (one weight per asset).
    - :class:`~fynance.core.protocols.CostModel`: ``__call__`` on a
      ``(T, N)`` weight path; must return a ``numpy.ndarray`` of shape
      ``(T,)`` (one cost per step).
    - :class:`~fynance.core.protocols.Metric`: ``__call__`` on a ``(T,)``
      return curve; must return a plain scalar (``int``/``float``/numpy
      scalar), *not* an array.
    - :class:`~fynance.core.protocols.DataSource`: **not smoke-run**. Its
      only method, ``load(*args, **kwargs)``, is an open-ended I/O port (the
      one boundary protocol that touches the outside world) -- there is no
      synthetic input this function can fabricate that would exercise a real
      adapter. Always raises :class:`ValueError` pointing at calling
      ``load()`` directly on a concrete instance instead.

    Any other object passed as `protocol` (not one of the six seams above)
    also raises :class:`ValueError`: there is no registered smoke recipe for
    it.

    Parameters
    ----------
    obj : object
        Candidate instance to check.
    protocol : type
        One of the ``typing.Protocol`` classes from
        :mod:`fynance.core.protocols`.
    T : int, optional
        Synthetic time length. Default is 64.
    N : int, optional
        Synthetic asset/feature count. Default is 3.
    seed : int, optional
        Seed of the synthetic data generator. Default is 0.

    Returns
    -------
    None
        Nothing on success.

    Raises
    ------
    AssertionError
        If `obj` does not structurally conform to `protocol`, or one of its
        methods returns a value of the wrong type/shape/dtype. The message
        names the offending method and the expected vs. actual type/shape.
    ValueError
        If `protocol` is :class:`~fynance.core.protocols.DataSource` (I/O
        port, see above), or is not one of the protocols this function knows
        how to smoke-test.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.core import FeatureTransform
    >>> from fynance.core.checks import check_conforms
    >>> class ZScore:
    ...     def fit(self, X):
    ...         self.mean_ = X.mean(axis=0)
    ...         self.std_ = X.std(axis=0)
    ...         return self
    ...     def transform(self, X):
    ...         return (X - self.mean_) / self.std_
    >>> check_conforms(ZScore(), FeatureTransform)
    >>> class BrokenTransform(ZScore):
    ...     def transform(self, X):
    ...         return list(X.mean(axis=0))
    >>> try:
    ...     check_conforms(BrokenTransform(), FeatureTransform)
    ... except AssertionError as e:
    ...     print(str(e))
    transform() returned list, expected numpy.ndarray

    """
    name = getattr(protocol, '__name__', repr(protocol))

    if protocol is DataSource:
        raise ValueError(
            "DataSource checks need a concrete instance and real load() "
            "arguments: load(*args, **kwargs) is an open-ended I/O port, so "
            "there is no synthetic input check_conforms can fabricate for "
            "it. Construct the concrete adapter and call load() directly in "
            "your own test instead."
        )

    if protocol not in _RECIPES:
        raise ValueError(
            f"no smoke-test recipe registered for protocol {name!r}; "
            f"check_conforms supports "
            f"{sorted(p.__name__ for p in _RECIPES)} (and raises ValueError "
            "for DataSource -- see its docstring)."
        )

    if not isinstance(obj, protocol):
        raise AssertionError(
            f"{obj!r} does not structurally conform to {name} (missing one "
            "or more of its required methods)."
        )

    rng = np.random.default_rng(seed)
    _RECIPES[protocol](obj, rng, T, N)

    return None


# =========================================================================== #
#                              Causality probe                                #
# =========================================================================== #


def _perturbations(
    x: NDArray, probe: int, rng: np.random.Generator,
) -> Iterator[tuple[str, NDArray]]:
    """ Yield ``(label, perturbed_x)`` pairs, perturbing `x` from `probe` onward. """
    noise_scale = 0.01 * float(np.abs(x[probe:]).mean())
    x_noise = x.copy()
    x_noise[probe:] += rng.standard_normal(x.shape[0] - probe) * noise_scale
    x_noise = np.clip(x_noise, 1e-8, None)  # keep levels positive (log-based features)
    yield 'additive noise', x_noise

    x_rescale = x.copy()
    x_rescale[probe:] *= 1.5
    yield '1.5x rescale', x_rescale


def _first_leak(y0: NDArray, y1: NDArray, limit: int, atol: float) -> int | None:
    """ Return the first index ``< limit`` where `y0` and `y1` disagree, or ``None``. """
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    n = min(y0.shape[0], y1.shape[0], limit)
    for i in range(n):
        if not np.allclose(y0[i], y1[i], atol=atol, rtol=0.0, equal_nan=True):
            return i

    return None


def assert_causal(
    func: Callable[[NDArray], NDArray],
    *,
    T: int = 256,
    t0: int | None = None,
    n_probes: int = 3,
    seed: int = 0,
    atol: float = 0.0,
) -> None:
    r""" Assert `func` never uses future information (no lookahead bias).

    Executable form of the repository's "no lookahead" house rule: generates
    a seeded synthetic price level series ``x``, computes the baseline
    ``y0 = func(x)``, then at one or more probe indices ``t0`` perturbs `x`
    strictly from ``t0`` onward -- once with additive noise, once with a
    ``1.5x`` rescale -- and requires ``func`` of the perturbed input to agree
    with ``y0`` everywhere *before* ``t0``. NaN is treated as equal to NaN
    (so a warmup region that is legitimately ``NaN`` does not trip the
    check).

    Parameters
    ----------
    func : callable
        Maps a 1-D array of length `T` to a 1-D or 2-D array aligned with it
        on axis 0 (e.g. a rolling feature). Extra keyword arguments should be
        bound with a ``lambda`` or :func:`functools.partial` before passing
        `func` in.
    T : int, optional
        Length of the synthetic input series. Default is 256.
    t0 : int, optional
        Single probe index to test. If ``None`` (default), `n_probes` probe
        points spread over ``[T // 4, 3 * T // 4]`` are used instead.
    n_probes : int, optional
        Number of default probe points when `t0` is ``None``. Default is 3
        (``T // 4``, ``T // 2``, ``3 * T // 4``).
    seed : int, optional
        Seed of the synthetic input and perturbation noise. Default is 0.
    atol : float, optional
        Absolute tolerance for the pre-``t0`` comparison. Default is 0.0
        (exact, NaN-safe equality).

    Returns
    -------
    None
        Nothing on success.

    Raises
    ------
    ValueError
        If `T` is too small to host a probe, or `t0` does not lie strictly
        inside ``(0, T - 1)``.
    AssertionError
        If perturbing the input from some probe `t0` onward changes the
        output before `t0`. The message names `func`, the probe `t0`, the
        perturbation used, and the earliest leaking index.

    Examples
    --------
    A trailing (causal) rolling mean passes:

    >>> import numpy as np
    >>> from fynance.core.checks import assert_causal
    >>> def trailing_mean(x, w=5):
    ...     out = np.empty_like(x)
    ...     for t in range(len(x)):
    ...         lo = max(0, t - w + 1)
    ...         out[t] = x[lo:t + 1].mean()
    ...     return out
    >>> assert_causal(trailing_mean, T=64, seed=0)

    A centered rolling mean (``mode='same'`` convolution) leaks future
    values into the past and raises, naming the earliest leaking index:

    >>> def centered_mean(x, w=9):
    ...     return np.convolve(x, np.ones(w) / w, mode='same')
    >>> try:
    ...     assert_causal(centered_mean, T=64, seed=0)
    ... except AssertionError as e:
    ...     print(str(e)[:40])
    assert_causal: lookahead detected in cen

    """
    if T < 8:
        raise ValueError(f"T must be at least 8 to host a probe, got {T}")

    rng = np.random.default_rng(seed)
    x = 100.0 * np.exp(np.cumsum(rng.standard_normal(T) * 0.01))

    y0 = func(x)

    if t0 is not None:
        if not (0 < t0 < T - 1):
            raise ValueError(f"t0={t0} must lie strictly inside (0, {T - 1})")
        probe_points = [int(t0)]

    else:
        lo, hi = T // 4, 3 * T // 4
        probe_points = sorted(
            {int(p) for p in np.linspace(lo, hi, num=max(n_probes, 1))}
        )
        probe_points = [p for p in probe_points if 0 < p < T - 1] or [T // 2]

    fname = getattr(func, '__name__', repr(func))

    for probe in probe_points:
        for label, x_pert in _perturbations(x, probe, rng):
            y1 = func(x_pert)
            leak = _first_leak(y0, y1, probe, atol)
            if leak is not None:
                raise AssertionError(
                    f"assert_causal: lookahead detected in {fname} -- "
                    f"perturbing the input from t0={probe} onward "
                    f"({label}) changed the output at index {leak} < t0; "
                    "output before t0 must depend only on inputs up to t0."
                )

    return None


if __name__ == '__main__':

    import doctest

    doctest.testmod()
