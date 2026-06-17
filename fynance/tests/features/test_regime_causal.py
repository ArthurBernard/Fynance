#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" I2 — causal `RegimeDetector` (fit-on-train / assign-online). """

# Third-party
import numpy as np

# Local
from fynance.features import RegimeDetector, detect_regimes, regime_features
from fynance.research import regime_switching


def _series(n=600, seed=1):
    return regime_switching(
        n, regimes=((0.0, 0.004), (0.0, 0.03)), p_switch=0.01, seed=seed
    ).to_numpy()


def test_regime_features_shape_and_causality():
    p = _series(300)
    f = regime_features(p, w=20)
    assert f.shape == (300, 2)
    # Causal: perturbing the tail leaves earlier feature rows untouched.
    pert = p.copy()
    pert[200:] *= 1.5
    assert np.array_equal(regime_features(p, w=20)[:200],
                          regime_features(pert, w=20)[:200])


def test_detector_determinism_and_shapes():
    p = _series()
    a = RegimeDetector(n_regimes=2, w=20, seed=0).fit_predict(p)
    b = RegimeDetector(n_regimes=2, w=20, seed=0).fit_predict(p)

    assert a.shape == (len(p),)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) <= {0, 1}


def test_labels_ordered_by_volatility():
    p = _series(900, seed=3)
    det = RegimeDetector(n_regimes=3, w=21, seed=0).fit(p)
    labels = det.predict(p)
    vol = regime_features(p, w=21)[:, 0]

    present = [k for k in range(3) if np.any(labels == k)]
    means = [vol[labels == k].mean() for k in present]
    # Label index increases with mean volatility (calmest = 0).
    assert means == sorted(means)


def test_no_lookahead_fit_on_past_assign_online():
    # THE causality guarantee: fit on a prefix, then perturb the future — the
    # labels on the earlier (unperturbed) prefix must be identical.
    p = _series(600, seed=1)
    det = RegimeDetector(n_regimes=2, w=20, seed=0).fit(p[:400])

    base = det.predict(p)
    pert = p.copy()
    cut = 450
    rng = np.random.default_rng(0)
    pert[cut:] *= np.cumprod(1.0 + rng.standard_normal(len(p) - cut) * 0.05)
    moved = det.predict(pert)

    assert np.array_equal(base[:cut], moved[:cut])


def test_fit_predict_matches_in_sample_detect_regimes():
    # Fit-predict on the whole series ≈ the in-sample detect_regimes (nearest
    # centroid == the k-means assignment).
    p = _series(500, seed=2)
    a = RegimeDetector(n_regimes=3, w=21, seed=0).fit_predict(p)
    b = detect_regimes(p, n_regimes=3, w=21, seed=0)

    # Same regimes; the few % gap is k-means' own labels vs nearest-centroid
    # reassignment on borderline points (the detector uses the latter, which is
    # also what it must do out-of-sample).
    assert (a == b).mean() > 0.90
