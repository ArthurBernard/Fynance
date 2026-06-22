""" Tests for _RollingBasis walk-forward CV helpers. """

import numpy as np
import pytest
import torch
import torch.nn as nn

from fynance.models.mlp import MultiLayerPerceptron
from fynance.models.rolling import (
    CVResult,
    RollMultiLayerPerceptron,
    _RollingBasis,
)

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


# §5.6 purged walk-forward CV

class TestPurge:
    def test_fold_slices_purge_shrinks_train_end(self):
        rb = make_rb()
        purge = 5
        for (tr, te), (tr_p, te_p) in zip(rb._fold_slices(), rb._fold_slices(purge=purge)):
            assert tr_p.stop == tr.stop - purge
            assert tr_p.start == tr.start
            assert te_p == te  # test window unchanged

    def test_cross_validate_with_purge_runs(self):
        rb = make_rb()
        result = rb.cross_validate(model_factory, X_t, y_t, purge=3)
        assert result.oof_predictions.shape == (T, N_OUT)


# ---------------------------------------------------------------------------
# Causality: window-start guards (no negative-index / future-wrapping windows)
# ---------------------------------------------------------------------------

class TestWindowGuards:

    def test_roll_period_exceeding_train_period_raises(self):
        # r > n would push the in-sample eval window (last r bars of the
        # training window) outside the training window and, with a negative
        # t0, turn slices into negative indices wrapping to future bars.
        rb = _RollingBasis(X_t, y_t)
        with pytest.raises(ValueError, match="roll_period"):
            rb(train_period=10, test_period=5, roll_period=20)

    def test_negative_start_does_not_produce_negative_windows(self):
        # A negative start used to let t0 go negative -> slice(t-r, t) and
        # arange(t-n, t) became negative indices wrapping to the tail (future
        # leak). t0 must be clamped to >= 0.
        rb = _RollingBasis(X_t, y_t)
        rb(train_period=20, test_period=5, start=-8, roll_period=10)
        assert rb.t0 >= 0
        it = iter(rb)
        for eval_set, test_set in it:
            assert eval_set.start >= 0
            assert test_set.start >= 0
            # the rebuilt training index must never be negative
            assert rb.t_idx.min() >= 0

    def test_all_windows_non_negative_and_train_before_test(self):
        # Sweep the iterator: every train/eval/test window starts at a
        # non-negative index and the training window precedes the test window.
        rb = _RollingBasis(X_t, y_t)
        rb(train_period=TRAIN, test_period=TEST, roll_period=ROLL)
        it = iter(rb)
        for eval_set, test_set in it:
            train_start = rb.t - rb.n
            assert train_start >= 0
            assert eval_set.start >= 0
            assert test_set.start >= 0
            # training window [t-n, t) is entirely before the test window.
            assert rb.t <= test_set.start
            # eval window is the in-sample tail: inside the training window.
            assert eval_set.start >= train_start
            assert eval_set.stop <= rb.t


# ---------------------------------------------------------------------------
# Reported training loss scale (must not depend on batch size)
# ---------------------------------------------------------------------------

class TestTrainLossScale:

    def _make(self, batch_size):
        # Seed both RNGs (weight init + the t_idx shuffle in _training) so the
        # reported loss is deterministic.
        torch.manual_seed(0)
        np.random.seed(0)
        model = RollMultiLayerPerceptron(X_t, y_t, layers=[8])
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        model.set_roll_period(
            train_period=TRAIN, test_period=TEST, roll_period=ROLL,
            batch_size=batch_size, epochs=1,
        )
        return model

    def _one_train_step(self, model):
        np.random.seed(0)
        it = iter(model)
        next(it)
        model._training()
        return model.loss_train[model.i]

    def test_train_loss_on_single_batch_scale(self):
        # _training reports the mean over batches of per-batch mean losses.
        # That must be on the same (MSE) scale as a single full-batch loss,
        # not divided by the train length n (the old `/ n` bug, which made the
        # reported loss ~ n_batches times too small).
        model = self._make(batch_size=8)   # 5 batches over n=40
        reported = self._one_train_step(model)
        # Reference MSE on the same training window: O(1) for unit-variance
        # data. The old `/ n` scaling would push it to ~ O(1/40) instead.
        ref = float(((model.predict(model.X[model.t_idx]).numpy()
                      - model.y[model.t_idx].numpy()) ** 2).mean())
        assert reported == pytest.approx(ref, rel=0.6, abs=0.6)
        assert reported > ref / 3   # not collapsed by the 1/n factor

    def test_train_loss_scale_independent_of_batch_size(self):
        # Reported train loss for batch_size=8 (5 batches) and batch_size=40
        # (1 batch) must be on the same scale. The old `/ n` bug divided the
        # 5-batch sum by n=40 (not by 5), so the small-batch loss came out
        # several times smaller -- a batch-size-dependent, meaningless scale.
        ls = self._one_train_step(self._make(batch_size=8))
        lf = self._one_train_step(self._make(batch_size=TRAIN))
        assert 0.4 < ls / lf < 2.5


# ---------------------------------------------------------------------------
# _display_kpi reads the current iteration, not the last array slot
# ---------------------------------------------------------------------------

class TestDisplayKpi:

    def test_kpi_reports_current_iteration_loss(self, capsys):
        # _display_kpi must read loss_eval[self.i] / loss_test[self.i] -- the
        # current step -- not [-1] (the last array slot, which stays 0 until the
        # final iteration and would print the wrong, stale value mid-run).
        rb = make_rb()
        rb.loss_eval = np.zeros(5)
        rb.loss_test = np.zeros(5)
        rb.i = 1
        rb.loss_eval[rb.i] = 0.42   # current step value
        rb.loss_test[rb.i] = 0.73
        # last slot stays 0 -> if the code used [-1] the printout would show 0.0
        rb._display_kpi(t=rb.n + rb.s)
        out = capsys.readouterr().out
        assert '0.42' in out
        assert '0.73' in out


# ---------------------------------------------------------------------------
# run(): minimal walk-forward sanity (display off)
# ---------------------------------------------------------------------------

class TestRunWalkForward:

    def test_run_no_display_train_precedes_test(self):
        # A tiny seeded walk-forward run with both display paths off must
        # complete and log per-step losses; and every training window must
        # precede its test window (causality).
        torch.manual_seed(0)
        model = RollMultiLayerPerceptron(X_t, y_t, layers=[8])
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        model.set_roll_period(
            train_period=TRAIN, test_period=TEST, roll_period=ROLL, epochs=1,
        )
        # Record the (train_start, test_start) pairs as the loop advances.
        bounds = []
        model.set_roll_period(
            train_period=TRAIN, test_period=TEST, roll_period=ROLL, epochs=1,
        )
        for eval_set, test_set in iter(model):
            bounds.append((model.t - model.n, test_set.start))
        for train_start, test_start in bounds:
            assert train_start >= 0
            assert train_start + TRAIN <= test_start

        model.set_roll_period(
            train_period=TRAIN, test_period=TEST, roll_period=ROLL, epochs=1,
        )
        out = model.run(backtest_plot=False, backtest_kpi=False)
        assert out is model
        stats = model.get_stats()
        assert stats.size > 0
        assert np.all(np.isfinite(stats["train_loss"]))
