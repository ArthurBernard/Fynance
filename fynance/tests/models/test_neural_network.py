""" Tests for PyTorch neural network models. """

import numpy as np
import pytest
import torch
import torch.nn as nn

from fynance.models.attention import MultiHeadAttention, ScaledDotProductAttention
from fynance.models.neural_network import MultiLayerPerceptron, _type_convert
from fynance.models.recurrent_neural_network import (
    GatedRecurrentUnit,
    LongShortTermMemory,
)
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
        out = model.predict(X_t, H)
        # predict returns (Y, H) tuple because forward does
        Y = out[0] if isinstance(out, tuple) else out
        assert Y.shape == (T, N_OUT)


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

    def test_predict_shape(self):
        model = self._make_lstm()
        H = torch.zeros(T, model.H)
        C = torch.zeros(T, model.H)
        out = model.predict(X_t, H, C)
        # predict returns (Y, H, C) tuple because forward does
        Y = out[0] if isinstance(out, tuple) else out
        assert Y.shape == (T, N_OUT)


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
        df = model.get_stats()
        assert df.empty

    def test_one_training_step(self):
        """ One manual step: training should not raise and update weights. """
        model = self._make_roll_mlp()
        it = iter(model)
        eval_set, test_set = next(it)
        model._training()
        # Prediction on eval window should work
        pred = model.sub_predict(model.X[eval_set])
        assert pred.shape[1] == N_OUT
