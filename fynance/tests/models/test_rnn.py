""" Tests for RecurrentNeuralNetwork. """

import numpy as np
import torch
import torch.nn as nn

from fynance.models.rnn import RecurrentNeuralNetwork

RNG = np.random.default_rng(0)

T, N_IN, N_OUT = 50, 6, 2
X_np = RNG.standard_normal((T, N_IN)).astype(np.float32)
y_np = RNG.standard_normal((T, N_OUT)).astype(np.float32)
X_t = torch.from_numpy(X_np)
y_t = torch.from_numpy(y_np)


def _make_rnn(hidden=12):
    model = RecurrentNeuralNetwork(X_t, y_t, hidden_state_size=hidden)
    model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
    return model


class TestRecurrentNeuralNetwork:

    def test_forward_returns_y_and_h(self):
        model = _make_rnn()
        H = torch.zeros(T, model.H)
        Y, H_out = model(X_t, H)
        assert Y.shape == (T, N_OUT)
        assert H_out.shape == (T, model.H)

    def test_train_on_returns_loss_and_state(self):
        model = _make_rnn()
        H = torch.zeros(T, model.H)
        loss, H_out = model.train_on(X_t, y_t, H)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        assert H_out.shape == (T, model.H)

    def test_predict_no_grad(self):
        model = _make_rnn()
        H = torch.zeros(T, model.H)
        Y, H_out = model.predict(X_t, H)
        assert not Y.requires_grad
        assert not H_out.requires_grad

    def test_int_constructor(self):
        model = RecurrentNeuralNetwork(N_IN, N_OUT, hidden_state_size=8)
        H = torch.zeros(T, model.H)
        Y, H_out = model(X_t, H)
        assert Y.shape == (T, N_OUT)

    def test_hidden_state_size_default(self):
        model = RecurrentNeuralNetwork(X_t, y_t)
        assert model.H == N_IN

    def test_hidden_state_size_custom(self):
        model = RecurrentNeuralNetwork(X_t, y_t, hidden_state_size=32)
        assert model.H == 32
        H = torch.zeros(T, 32)
        Y, H_out = model(X_t, H)
        assert H_out.shape == (T, 32)
