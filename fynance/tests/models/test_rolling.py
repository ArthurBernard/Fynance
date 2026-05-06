""" Tests for _RollingBasis walk-forward CV helpers. """

import numpy as np
import pytest
import torch
import torch.nn as nn

from fynance.models.mlp import MultiLayerPerceptron
from fynance.models.rolling import CVResult, _RollingBasis

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
T, N_IN, N_OUT = 80, 4, 1

X_np = RNG.standard_normal((T, N_IN)).astype(np.float32)
y_np = RNG.standard_normal((T, N_OUT)).astype(np.float32)
X_t = torch.from_numpy(X_np)
y_t = torch.from_numpy(y_np)

TRAIN, TEST, ROLL = 40, 10, 10


def make_rb():
    rb = _RollingBasis(X_t, y_t)
    rb(train_period=TRAIN, test_period=TEST, roll_period=ROLL)
    return rb


def model_factory():
    m = MultiLayerPerceptron(N_IN, N_OUT, layers=[8])
    m.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
    return m


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


# ---------------------------------------------------------------------------
# _fold_slices
# ---------------------------------------------------------------------------

class TestFoldSlices:

    def test_fold_count(self):
        rb = make_rb()
        folds = list(rb._fold_slices())
        expected = (T - rb.t0 - TEST) // ROLL
        assert len(folds) == expected

    def test_no_overlap(self):
        rb = make_rb()
        slices = [test_sl for _, test_sl in rb._fold_slices()]
        for a, b in zip(slices, slices[1:]):
            assert a.stop <= b.start

    def test_train_size(self):
        rb = make_rb()
        for train_sl, _ in rb._fold_slices():
            assert train_sl.stop - train_sl.start == TRAIN

    def test_temporal_order(self):
        rb = make_rb()
        for train_sl, test_sl in rb._fold_slices():
            assert train_sl.stop <= test_sl.start


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------

class TestCrossValidate:

    def _run(self, metric_fn=None):
        rb = make_rb()
        return rb.cross_validate(model_factory, X_t, y_t, metric_fn=metric_fn)

    def test_returns_cvresult(self):
        result = self._run()
        assert isinstance(result, CVResult)

    def test_oof_shape(self):
        result = self._run()
        assert result.oof_predictions.shape == (T, N_OUT)

    def test_oof_nan_before_first_fold(self):
        rb = make_rb()
        result = rb.cross_validate(model_factory, X_t, y_t)
        assert np.all(np.isnan(result.oof_predictions[: rb.t0]))

    def test_fold_metrics_length(self):
        rb = make_rb()
        folds = list(rb._fold_slices())
        result = rb.cross_validate(model_factory, X_t, y_t, metric_fn=mse)
        assert len(result.fold_metrics) == len(folds)

    def test_no_metric_fn_gives_none(self):
        result = self._run(metric_fn=None)
        assert result.fold_metrics == []
        assert result.mean_metric is None
        assert result.std_metric is None

    def test_mean_metric_consistent(self):
        result = self._run(metric_fn=mse)
        assert result.mean_metric == pytest.approx(np.mean(result.fold_metrics))

    def test_std_metric_consistent(self):
        result = self._run(metric_fn=mse)
        assert result.std_metric == pytest.approx(np.std(result.fold_metrics))
