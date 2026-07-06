#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :mod:`fynance.research.bootstrap`. """

# Third-party
import numpy as np
import pytest

# Local
from fynance.research import (
    block_permutation_test,
    bootstrap_metric,
    resample_paths,
)

# -- resample_paths: shapes / dtypes / reproducibility ----------------------

@pytest.mark.parametrize("method", ["circular", "stationary"])
def test_resample_paths_shape_and_dtype(method):
    r = np.random.default_rng(0).standard_normal(50)
    paths = resample_paths(r, n_paths=17, block=5, method=method, seed=0)

    assert paths.shape == (17, 50)
    assert paths.dtype == np.float64


@pytest.mark.parametrize("method", ["circular", "stationary"])
def test_resample_paths_reproducible(method):
    r = np.random.default_rng(1).standard_normal(80)
    a = resample_paths(r, n_paths=10, block=7, method=method, seed=42)
    b = resample_paths(r, n_paths=10, block=7, method=method, seed=42)

    assert np.array_equal(a, b)


def test_resample_paths_different_seeds_differ():
    r = np.random.default_rng(1).standard_normal(80)
    a = resample_paths(r, n_paths=10, block=7, seed=1)
    b = resample_paths(r, n_paths=10, block=7, seed=2)

    assert not np.array_equal(a, b)


@pytest.mark.parametrize("kwargs, match", [
    ({"returns": [1.0]}, "at least 2 observations"),
    ({"returns": np.arange(10.0), "n_paths": 0}, "n_paths"),
    ({"returns": np.arange(10.0), "block": 0}, "block"),
    ({"returns": np.arange(10.0), "method": "bogus"}, "method"),
])
def test_resample_paths_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        resample_paths(**kwargs)


# -- circular block bootstrap: structural check ------------------------------

def test_circular_blocks_are_contiguous_original_segments():
    # Values equal their own original index, so each block of the output can
    # be checked against the actual wrap-around segment it was drawn from.
    T, block = 12, 4
    r = np.arange(T, dtype=np.float64)
    paths = resample_paths(r, n_paths=5, block=block, method="circular", seed=2)

    for path in paths:
        for b in range(T // block):
            seg = path[b * block:(b + 1) * block]
            start = int(seg[0])
            expected = [(start + j) % T for j in range(block)]
            assert seg.tolist() == expected


# -- stationary block bootstrap: geometric block length -----------------------

def test_stationary_mean_block_length_within_tolerance():
    # Values equal their own original index again, so consecutive index jumps
    # of exactly 1 (mod T) reveal a block continuation; anything else marks a
    # new block start. Run-length encoding those starts over many paths
    # recovers the empirical mean block length, which should be close to the
    # geometric distribution's mean (`block`).
    T, block = 500, 15
    r = np.arange(T, dtype=np.float64)
    paths = resample_paths(r, n_paths=200, block=block, method="stationary", seed=7)
    idx = np.round(paths).astype(np.int64)

    lengths = []
    for path in idx:
        jumps = (np.diff(path) - 1) % T
        new_start = np.concatenate([[True], jumps != 0])
        starts = np.flatnonzero(new_start)
        run_lengths = np.diff(np.concatenate([starts, [T]]))
        lengths.extend(run_lengths[:-1])  # drop the last (possibly truncated) block

    mean_length = np.mean(lengths)
    assert abs(mean_length - block) / block < 0.20


# -- bootstrap_metric --------------------------------------------------------

def test_bootstrap_metric_structure_and_reproducible():
    r = np.random.default_rng(0).standard_normal(100)
    out1 = bootstrap_metric(r, np.mean, n_paths=200, block=5, seed=3)
    out2 = bootstrap_metric(r, np.mean, n_paths=200, block=5, seed=3)

    assert set(out1) == {"estimate", "lo", "hi", "distribution"}
    assert out1["estimate"] == pytest.approx(float(np.mean(r)))
    assert out1["lo"] <= out1["hi"]
    assert out1["distribution"].shape == (200,)
    assert out1["estimate"] == out2["estimate"]
    assert out1["lo"] == out2["lo"] and out1["hi"] == out2["hi"]
    assert np.array_equal(out1["distribution"], out2["distribution"])


def test_bootstrap_metric_invalid_ci():
    r = np.arange(10.0)
    with pytest.raises(ValueError, match="ci"):
        bootstrap_metric(r, np.mean, ci=1.5)


def test_bootstrap_metric_iid_normal_coverage():
    # Over many independent repetitions on i.i.d. normal data (mean 0), a 90%
    # CI should cover the true mean roughly 90% of the time; require at least
    # 75% to keep the test robust to sampling noise while still catching a
    # badly miscalibrated interval. Kept small (T, n_paths) to stay fast.
    n_reps = 200
    T = 60
    hits = 0
    for rep in range(n_reps):
        data = np.random.default_rng(1000 + rep).standard_normal(T)
        out = bootstrap_metric(data, np.mean, n_paths=300, block=5,
                               method="stationary", ci=0.9, seed=rep)
        if out["lo"] <= 0.0 <= out["hi"]:
            hits += 1

    assert hits / n_reps >= 0.75


def test_bootstrap_metric_ar1_ci_wider_than_iid():
    # AR(1, phi=0.5) returns: the block bootstrap (preserving autocorrelation)
    # must give a strictly wider CI for the mean than a naive i.i.d. resample
    # (which ignores the dependence and so understates the estimator's true
    # variance).
    T, phi = 500, 0.5
    rng = np.random.default_rng(3)
    r = np.zeros(T)
    for t in range(1, T):
        r[t] = phi * r[t - 1] + rng.standard_normal()

    block_out = bootstrap_metric(r, np.mean, n_paths=500, block=20,
                                 method="stationary", seed=0)
    width_block = block_out["hi"] - block_out["lo"]

    # Inline i.i.d. resample comparison (rng.choice-based bootstrap).
    rng_iid = np.random.default_rng(0)
    iid_dist = np.empty(500)
    for i in range(500):
        sample = rng_iid.choice(r, size=T, replace=True)
        iid_dist[i] = np.mean(sample)
    lo_iid, hi_iid = np.quantile(iid_dist, [0.025, 0.975])
    width_iid = hi_iid - lo_iid

    assert width_block > width_iid


# -- block_permutation_test --------------------------------------------------

def test_block_permutation_test_driftless_noise():
    driftless = 0.01 * np.random.default_rng(8).standard_normal(600)
    p = block_permutation_test(driftless, n_perm=500, block=20, seed=0)

    assert 0.05 < p < 0.95


def test_block_permutation_test_strong_drift():
    drifted = 0.002 + 0.01 * np.random.default_rng(2008).standard_normal(600)
    p = block_permutation_test(drifted, n_perm=500, block=20, seed=0)

    assert p < 0.05


def test_block_permutation_test_reproducible():
    r = np.random.default_rng(4).standard_normal(300)
    p1 = block_permutation_test(r, n_perm=200, block=10, seed=5)
    p2 = block_permutation_test(r, n_perm=200, block=10, seed=5)

    assert p1 == p2
    assert 0.0 < p1 <= 1.0
