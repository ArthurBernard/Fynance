""" Tests for PyTorch neural network models. """

import numpy as np
import pytest
import torch
import torch.nn as nn

from fynance.models._base import _type_convert
from fynance.models.attention import MultiHeadAttention, ScaledDotProductAttention
from fynance.models.gru import GatedRecurrentUnit, GRUCell
from fynance.models.loss import SharpeLoss
from fynance.models.lstm import LongShortTermMemory, LSTMCell
from fynance.models.mlp import MultiLayerPerceptron
from fynance.models.rolling import RollMultiLayerPerceptron

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

T, N_IN, N_OUT = 100, 8, 2
X_np = RNG.standard_normal((T, N_IN)).astype(np.float32)
y_np = RNG.standard_normal((T, N_OUT)).astype(np.float32)
X_t = torch.from_numpy(X_np)
y_t = torch.from_numpy(y_np)


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------

class TestTypeConvert:

    def test_float64_mapping(self):
        assert _type_convert(np.float64) == torch.float64

    def test_float32_mapping(self):
        assert _type_convert(np.float32) == torch.float32

    def test_float16_mapping(self):
        assert _type_convert(np.float16) == torch.float16

    def test_uint8_mapping(self):
        assert _type_convert(np.uint8) == torch.uint8

    def test_int8_mapping(self):
        assert _type_convert(np.int8) == torch.int8

    def test_int16_mapping(self):
        assert _type_convert(np.int16) == torch.int16

    def test_int32_mapping(self):
        assert _type_convert(np.int32) == torch.int32

    def test_int64_mapping(self):
        assert _type_convert(np.int64) == torch.int64

    def test_unknown_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unkwnown type"):
            _type_convert(object())


# ---------------------------------------------------------------------------
# MultiLayerPerceptron
# ---------------------------------------------------------------------------

class TestMLP:

    def _make_mlp(self):
        model = MultiLayerPerceptron(X_t, y_t, layers=[16, 8])
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        return model

    def test_forward_shape(self):
        model = self._make_mlp()
        out = model(X_t)
        assert out.shape == (T, N_OUT)

    def test_train_on_returns_loss(self):
        model = self._make_mlp()
        loss = model.train_on(X_t, y_t)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_predict_shape(self):
        model = self._make_mlp()
        pred = model.predict(X_t)
        assert pred.shape == (T, N_OUT)

    def test_predict_no_grad(self):
        model = self._make_mlp()
        pred = model.predict(X_t)
        assert not pred.requires_grad

    def test_set_data_validates_columns(self):
        model = self._make_mlp()
        bad_X = torch.randn(T, N_IN + 1)
        with pytest.raises(ValueError):
            model.set_data(bad_X, y_t)

    def test_int_constructor(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16])
        out = model(X_t)
        assert out.shape == (T, N_OUT)

    def test_single_activation(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16], activation=nn.ReLU)
        out = model(X_t)
        assert out.shape == (T, N_OUT)

    def test_list_activation_valid(self):
        model = MultiLayerPerceptron(
            N_IN, N_OUT, layers=[16, 8],
            activation=[nn.ReLU, nn.ReLU, nn.ReLU],
        )
        out = model(X_t)
        assert out.shape == (T, N_OUT)

    def test_list_activation_wrong_length_raises(self):
        with pytest.raises(ValueError):
            MultiLayerPerceptron(
                N_IN, N_OUT, layers=[16],
                activation=[nn.ReLU, nn.ReLU, nn.ReLU],
            )

    def test_scalar_drop(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16], drop=0.1)
        out = model(X_t)
        assert out.shape == (T, N_OUT)

    def test_list_drop_valid(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16], drop=[0.1, 0.2])
        out = model(X_t)
        assert out.shape == (T, N_OUT)

    def test_list_drop_wrong_length_raises(self):
        with pytest.raises(ValueError):
            MultiLayerPerceptron(N_IN, N_OUT, layers=[16], drop=[0.1, 0.2, 0.3])

    def test_set_lr_scheduler(self):
        model = self._make_mlp()
        model.set_lr_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10)
        assert model.lr_scheduler is not None

    def test_set_lr_scheduler_no_optimizer_raises(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16])
        with pytest.raises(ValueError):
            model.set_lr_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10)

    def test_train_on_with_lr_scheduler(self):
        model = self._make_mlp()
        model.set_lr_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10)
        loss = model.train_on(X_t, y_t)
        assert loss.item() >= 0

    def test_set_data_wrong_m_raises(self):
        model = self._make_mlp()
        bad_y = torch.randn(T, N_OUT + 1)
        with pytest.raises(ValueError):
            model.set_data(X_t, bad_y)

    def test_set_data_mismatched_length_raises(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16])
        bad_X = torch.randn(T + 5, N_IN)
        with pytest.raises(ValueError):
            model.set_data(bad_X, y_t)

    def test_set_seed_with_int(self):
        model = self._make_mlp()
        model.set_seed(seed_torch=42, seed_numpy=7)
        assert model.seed_torch == 42
        assert model.seed_numpy == 7

    def test_set_data_from_polars(self):
        import polars as pl
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16])
        X_df = pl.DataFrame(X_np)
        t = model._set_data(X_df)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (T, N_IN)

    def test_set_data_unknown_type_raises(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16])
        with pytest.raises(ValueError, match="Unkwnown data type"):
            model._set_data([1, 2, 3])

    def test_save_load_model_with_optimizer(self, tmp_path):
        model = self._make_mlp()
        path = tmp_path / "model.pt"
        model.save_model(path, save_optimizer=True)
        model2 = MultiLayerPerceptron(N_IN, N_OUT, layers=[16, 8])
        model2.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        model2.load_model(path, load_optimizer=True)
        assert model2.N == N_IN

    def test_load_model_no_optimizer_in_file_raises(self, tmp_path):
        model = self._make_mlp()
        path = tmp_path / "model_no_opt.pt"
        model.save_model(path, save_optimizer=False)
        model2 = MultiLayerPerceptron(N_IN, N_OUT, layers=[16, 8])
        model2.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        with pytest.raises(ValueError, match="No optimizer available"):
            model2.load_model(path, load_optimizer=True)

    def test_set_optimizer_with_module_list(self):
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[16, 8])
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, params=[model], lr=1e-3)
        assert model.optimizer is not None

    def test_dtype_request_is_applied(self):
        """ A float32 request on float64 data yields float32 tensors. """
        X64 = np.zeros((10, N_IN), dtype=np.float64)
        y64 = np.zeros((10, N_OUT), dtype=np.float64)
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[8])
        model.set_data(X64, y64, x_type=torch.float32, y_type=torch.float32)
        assert model.X.dtype == torch.float32
        assert model.y.dtype == torch.float32

    def test_float64_numpy_trains_without_astype(self):
        """ Plain float64 numpy input must fit/predict without manual cast. """
        rng = np.random.default_rng(0)
        X64 = rng.standard_normal((20, N_IN))  # float64 by default
        y64 = rng.standard_normal((20, N_OUT))
        assert X64.dtype == np.float64
        model = MultiLayerPerceptron(X64, y64, layers=[8])
        model.set_optimizer(nn.MSELoss, torch.optim.SGD, lr=1e-3)
        # Coerced to the default (float32) so it matches the float32 params.
        assert model.X.dtype == torch.get_default_dtype()
        loss = model.train_on(model.X, model.y)  # must not raise on dtype
        assert torch.isfinite(loss)
        pred = model.predict(model.X)
        assert pred.shape == (20, N_OUT)

    def test_set_data_does_not_alias_numpy(self):
        """ set_data must not let later edits to the tensor mutate the source. """
        X = np.zeros((10, N_IN), dtype=np.float32)
        model = MultiLayerPerceptron(N_IN, N_OUT, layers=[8])
        t = model._set_data(X)
        t[0, 0] = 123.0
        assert X[0, 0] == 0.0

    def test_predict_deterministic_with_dropout(self):
        """ With dropout>0, predict is deterministic and restores train mode. """
        torch.manual_seed(0)
        model = MultiLayerPerceptron(X_t, y_t, layers=[16], drop=0.5)
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        assert model.training  # starts in train mode
        p1 = model.predict(X_t)
        p2 = model.predict(X_t)
        assert torch.allclose(p1, p2)
        # eval mode must be restored to training after predict
        assert model.training

    def test_train_on_sets_train_mode(self):
        """ train_on must put the model back into training mode. """
        model = MultiLayerPerceptron(X_t, y_t, layers=[16], drop=0.5)
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        model.eval()
        model.train_on(X_t, y_t)
        assert model.training

    def test_overfit_tiny_batch_loss_decreases(self):
        """ Sanity: MLP can overfit a tiny batch (loss decreases). """
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        X = torch.from_numpy(rng.standard_normal((8, N_IN)).astype(np.float32))
        y = torch.from_numpy(rng.standard_normal((8, N_OUT)).astype(np.float32))
        model = MultiLayerPerceptron(X, y, layers=[32, 32], activation=nn.ReLU)
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
        first = model.train_on(X, y).item()
        for _ in range(200):
            last = model.train_on(X, y).item()
        assert last < first


# ---------------------------------------------------------------------------
# GatedRecurrentUnit
# ---------------------------------------------------------------------------

class TestGRU:

    def _make_gru(self, hidden=16):
        return GatedRecurrentUnit(X_t, y_t, hidden_state_size=hidden)

    def test_forward_shape(self):
        model = self._make_gru()
        H = torch.zeros(T, model.H)
        Y, H_out = model(X_t, H)
        assert Y.shape == (T, N_OUT)
        assert H_out.shape == (T, model.H)

    def test_train_on_returns_loss(self):
        model = self._make_gru()
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        H = torch.zeros(T, model.H)
        loss, _ = model.train_on(X_t, y_t, H)
        assert loss.item() >= 0

    def test_predict_shape(self):
        model = self._make_gru()
        H = torch.zeros(T, model.H)
        Y, H_out = model.predict(X_t, H)
        assert Y.shape == (T, N_OUT)
        assert H_out.shape == (T, model.H)

    def test_predict_no_grad(self):
        model = self._make_gru()
        H = torch.zeros(T, model.H)
        Y, H_out = model.predict(X_t, H)
        assert not Y.requires_grad
        assert not H_out.requires_grad

    def test_int_constructor(self):
        model = GatedRecurrentUnit(N_IN, N_OUT, hidden_state_size=8)
        H = torch.zeros(T, model.H)
        Y, H_out = model(X_t, H)
        assert Y.shape == (T, N_OUT)

    def test_hidden_state_size_default(self):
        model = GatedRecurrentUnit(X_t, y_t)
        assert model.H == N_IN

    def test_bias_false_removes_biases(self):
        """ bias=False must drop the bias on every linear layer. """
        model = GatedRecurrentUnit(N_IN, N_OUT, hidden_state_size=8, bias=False)
        assert model.W_h.bias is None
        assert model.W_u.bias is None
        assert model.W_r.bias is None
        assert model.W_y.bias is None
        # default keeps biases
        model_b = GatedRecurrentUnit(N_IN, N_OUT, hidden_state_size=8)
        assert model_b.W_h.bias is not None
        assert model_b.W_y.bias is not None

    def test_default_activation_is_not_simplex(self):
        """ Default output must not be a probability simplex (no Softmax). """
        torch.manual_seed(0)
        model = GatedRecurrentUnit(N_IN, N_OUT, hidden_state_size=8)
        H = torch.zeros(T, model.H)
        Y, _ = model(X_t, H)
        row_sums = Y.sum(dim=-1)
        # a Softmax default would force every row sum to exactly 1
        assert not torch.allclose(row_sums, torch.ones(T))


# ---------------------------------------------------------------------------
# LongShortTermMemory
# ---------------------------------------------------------------------------

class TestLSTM:

    def _make_lstm(self, hidden=16):
        return LongShortTermMemory(X_t, y_t, hidden_state_size=hidden)

    def test_forward_shape(self):
        model = self._make_lstm()
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        Y, H_out, C_out = model(X_t, H, C)
        assert Y.shape == (T, N_OUT)
        assert H_out.shape == (T, model.H)
        assert C_out.shape == (T, model.H)

    def test_train_on_returns_loss_and_states(self):
        model = self._make_lstm()
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        loss, H_out, C_out = model.train_on(X_t, y_t, H, C)
        assert loss.item() >= 0
        assert H_out.shape == (T, model.H)
        assert C_out.shape == (T, model.H)

    def test_predict_shape(self):
        model = self._make_lstm()
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        Y, H_out, C_out = model.predict(X_t, H, C)
        assert Y.shape == (T, N_OUT)
        assert H_out.shape == (T, model.H)
        assert C_out.shape == (T, model.H)

    def test_predict_no_grad(self):
        model = self._make_lstm()
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        Y, H_out, C_out = model.predict(X_t, H, C)
        assert not Y.requires_grad
        assert not H_out.requires_grad
        assert not C_out.requires_grad

    def test_int_constructor(self):
        model = LongShortTermMemory(N_IN, N_OUT, hidden_state_size=8)
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        Y, H_out, C_out = model(X_t, H, C)
        assert Y.shape == (T, N_OUT)

    def test_hidden_state_size_default(self):
        model = LongShortTermMemory(X_t, y_t)
        assert model.H == N_IN

    def test_bias_false_removes_biases(self):
        """ bias=False must drop the bias on every linear layer. """
        model = LongShortTermMemory(
            N_IN, N_OUT, hidden_state_size=8, bias=False
        )
        for w in (model.W_f, model.W_i, model.W_c, model.W_o,
                  model.W_h, model.W_y):
            assert w.bias is None

    def test_default_activation_is_not_simplex(self):
        """ Default output must not be a probability simplex (no Softmax). """
        torch.manual_seed(0)
        model = LongShortTermMemory(N_IN, N_OUT, hidden_state_size=8)
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        Y, _, _ = model(X_t, H, C)
        assert not torch.allclose(Y.sum(dim=-1), torch.ones(T))

    def test_predict_deterministic_with_dropout(self):
        """ predict with dropout>0 is deterministic and restores train mode. """
        torch.manual_seed(0)
        model = LongShortTermMemory(X_t, y_t, hidden_state_size=8, drop=0.5)
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        Y1, _, _ = model.predict(X_t, H, C)
        Y2, _, _ = model.predict(X_t, H, C)
        assert torch.allclose(Y1, Y2)
        assert model.training


# ---------------------------------------------------------------------------
# GRUCell
# ---------------------------------------------------------------------------

class TestGRUCell:

    def test_forward_shape(self):
        cell = GRUCell(N_IN, hidden_state_size=16)
        H = torch.zeros(T, 16)
        H_new = cell(X_t, H)
        assert H_new.shape == (T, 16)

    def test_forward_shape_int_only(self):
        cell = GRUCell(8, hidden_state_size=16)
        H = torch.zeros(T, 16)
        X = torch.randn(T, 8)
        H_new = cell(X, H)
        assert H_new.shape == (T, 16)

    def test_train_on_raises(self):
        cell = GRUCell(N_IN, hidden_state_size=16)
        with pytest.raises(NotImplementedError):
            cell.train_on(X_t, y_t, torch.zeros(T, 16))

    def test_predict_raises(self):
        cell = GRUCell(N_IN, hidden_state_size=16)
        with pytest.raises(NotImplementedError):
            cell.predict(X_t, torch.zeros(T, 16))

    def test_full_model_train_still_works(self):
        model = GatedRecurrentUnit(X_t, y_t, hidden_state_size=16)
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        H = torch.zeros(T, model.H)
        loss, _ = model.train_on(X_t, y_t, H)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# LSTMCell
# ---------------------------------------------------------------------------

class TestLSTMCell:

    def test_forward_shape(self):
        cell = LSTMCell(N_IN, hidden_state_size=16)
        H = torch.zeros(T, 16)
        C = torch.zeros(T, 16)
        H_new, C_new = cell(X_t, H, C)
        assert H_new.shape == (T, 16)
        assert C_new.shape == (T, 16)

    def test_train_on_raises(self):
        cell = LSTMCell(N_IN, hidden_state_size=16)
        with pytest.raises(NotImplementedError):
            cell.train_on(X_t, y_t, torch.zeros(T, 16), torch.zeros(T, 16))

    def test_predict_raises(self):
        cell = LSTMCell(N_IN, hidden_state_size=16)
        with pytest.raises(NotImplementedError):
            cell.predict(X_t, torch.zeros(T, 16), torch.zeros(T, 16))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttention:

    def test_sdpa_shape(self):
        attn = ScaledDotProductAttention()
        B, T_seq, d_k, d_v = 4, 10, 16, 32
        Q = torch.randn(B, T_seq, d_k)
        K = torch.randn(B, T_seq, d_k)
        V = torch.randn(B, T_seq, d_v)
        out, weights = attn(Q, K, V)
        assert out.shape == (B, T_seq, d_v)
        assert weights.shape == (B, T_seq, T_seq)

    def test_sdpa_weights_sum_to_one(self):
        attn = ScaledDotProductAttention()
        Q = K = V = torch.randn(2, 5, 8)
        _, weights = attn(Q, K, V)
        np.testing.assert_allclose(
            weights.sum(dim=-1).numpy(), np.ones((2, 5)), atol=1e-5
        )

    def test_mha_shape(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out, attn = mha(x)
        assert out.shape == (2, 10, 64)
        assert attn.shape == (2, 4, 10, 10)

    def test_mha_residual_norm(self):
        """ Output should differ from input (residual + norm applied). """
        torch.manual_seed(0)
        mha = MultiHeadAttention(d_model=32, num_heads=4)
        x = torch.randn(1, 6, 32)
        out, _ = mha(x)
        assert not torch.allclose(out, x)

    def test_mha_invalid_heads(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=33, num_heads=4)

    def test_sdpa_mask_zeros_weight(self):
        """ Masked positions (mask == 0) get ~0 attention weight. """
        torch.manual_seed(0)
        attn = ScaledDotProductAttention()
        B, T_seq, d = 2, 4, 8
        Q = torch.randn(B, T_seq, d)
        K = torch.randn(B, T_seq, d)
        V = torch.randn(B, T_seq, d)
        mask = torch.ones(B, T_seq, T_seq)
        mask[:, :, -1] = 0  # forbid attending to the last key position
        _, weights = attn(Q, K, V, mask=mask)
        # weight on masked positions must be (near) zero
        assert torch.allclose(weights[:, :, -1], torch.zeros(B, T_seq), atol=1e-6)
        # unmasked rows still sum to one
        np.testing.assert_allclose(
            weights.sum(dim=-1).numpy(), np.ones((B, T_seq)), atol=1e-5
        )

    def test_mha_mask_zeros_weight(self):
        """ MultiHeadAttention honours the mask: masked keys get ~0 weight. """
        torch.manual_seed(0)
        mha = MultiHeadAttention(d_model=16, num_heads=2)
        x = torch.randn(2, 5, 16)
        mask = torch.ones(2, 1, 5, 5)
        mask[:, :, :, -1] = 0  # forbid attending to the last position
        _, attn = mha(x, mask=mask)
        assert torch.allclose(
            attn[:, :, :, -1], torch.zeros(2, mha.num_heads, 5), atol=1e-6
        )


# ---------------------------------------------------------------------------
# RollMultiLayerPerceptron
# ---------------------------------------------------------------------------

class TestRollMLP:

    def _make_roll_mlp(self):
        model = RollMultiLayerPerceptron(X_t, y_t, layers=[16])
        model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
        model.set_roll_period(
            train_period=40, test_period=10, roll_period=10, epochs=1
        )
        return model

    def test_iteration_yields_slices(self):
        model = self._make_roll_mlp()
        steps = list(iter(model))
        assert len(steps) > 0
        eval_set, test_set = steps[0]
        assert isinstance(eval_set, slice)
        assert isinstance(test_set, slice)

    def test_get_stats_empty_before_run(self):
        model = self._make_roll_mlp()
        stats = model.get_stats()
        assert stats.size == 0

    def test_one_training_step(self):
        """ One manual step: training should not raise and update weights. """
        model = self._make_roll_mlp()
        it = iter(model)
        eval_set, test_set = next(it)
        model._training()
        # Prediction on eval window should work
        pred = model.sub_predict(model.X[eval_set])
        assert pred.shape[1] == N_OUT

    def test_get_stats_populated(self):
        model = self._make_roll_mlp()
        model.log = [
            {"step": 0, "train_loss": 1.0, "eval_loss": 2.0, "test_loss": 3.0},
            {"step": 1, "train_loss": 0.5, "eval_loss": 1.5, "test_loss": 2.5},
        ]
        stats = model.get_stats()
        assert stats.dtype.names == (
            "step", "train_loss", "eval_loss", "test_loss"
        )
        assert stats.size == 2
        assert stats["step"].tolist() == [0, 1]
        assert np.isclose(stats["train_loss"][1], 0.5)

    def test_training_updates_loss_train(self):
        model = self._make_roll_mlp()
        it = iter(model)
        next(it)
        model._training()
        assert np.isfinite(model.loss_train[model.i])


class TestRollMLPWithSharpeLoss:
    """ Integration: train a RollMLP with the differentiable SharpeLoss. """

    def _make(self):
        model = RollMultiLayerPerceptron(X_t, y_t, layers=[16])
        # SharpeLoss is passed as the criterion class; set_optimizer does
        # criterion() and the training loop calls criterion(outputs, y).
        model.set_optimizer(SharpeLoss, torch.optim.Adam, lr=1e-2)
        model.set_roll_period(
            train_period=40, test_period=10, roll_period=10, epochs=1
        )
        return model

    def test_training_step_runs_and_is_finite(self):
        model = self._make()
        it = iter(model)
        next(it)
        model._training()
        assert np.isfinite(model.loss_train[model.i])

    def test_optimizer_updates_weights(self):
        model = self._make()
        before = [p.detach().clone() for p in model.parameters()]
        it = iter(model)
        next(it)
        for _ in range(3):  # a few epochs on the first train window
            model._training()
        after = list(model.parameters())
        # at least one parameter tensor must have moved under SharpeLoss grads
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_prediction_shape_after_training(self):
        model = self._make()
        it = iter(model)
        eval_set, _ = next(it)
        model._training()
        pred = model.sub_predict(model.X[eval_set])
        assert pred.shape[1] == N_OUT
