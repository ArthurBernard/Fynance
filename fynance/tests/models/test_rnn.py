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

    def test_bias_false_removes_biases(self):
        """ bias=False must drop the bias on the recurrent and output layers. """
        model = RecurrentNeuralNetwork(N_IN, N_OUT, hidden_state_size=8, bias=False)
        assert model.W_h.bias is None
        assert model.W_y.bias is None
        model_b = RecurrentNeuralNetwork(N_IN, N_OUT, hidden_state_size=8)
        assert model_b.W_h.bias is not None
        assert model_b.W_y.bias is not None

    def test_default_output_is_not_simplex(self):
        """ Default forward_activation is Identity, not Softmax. """
        torch.manual_seed(0)
        model = RecurrentNeuralNetwork(N_IN, N_OUT, hidden_state_size=8)
        H = torch.zeros(T, model.H)
        Y, _ = model(X_t, H)
        # a Softmax default would force every row to sum to exactly 1
        assert not torch.allclose(Y.sum(dim=-1), torch.ones(T))

    def test_rows_processed_independently(self):
        """ Honest stateless contract: each row depends only on its own row.

        These cells do NOT thread state across the leading dimension, so
        perturbing one row of (X, H) must leave the other rows' outputs
        unchanged. This pins the documented stateless behaviour.
        """
        torch.manual_seed(0)
        model = RecurrentNeuralNetwork(N_IN, N_OUT, hidden_state_size=5)
        model.eval()
        X = torch.randn(7, N_IN)
        H = torch.randn(7, model.H)
        with torch.no_grad():
            Y_base, H_base = model(X, H)
            X_pert = X.clone()
            X_pert[3] += 100.0  # perturb a single row only
            Y_pert, H_pert = model(X_pert, H)
        idx = [i for i in range(7) if i != 3]
        assert torch.allclose(Y_base[idx], Y_pert[idx], atol=1e-5)
        assert torch.allclose(H_base[idx], H_pert[idx], atol=1e-5)
        # the perturbed row itself does change
        assert not torch.allclose(Y_base[3], Y_pert[3])

    def test_fit_predict_signalmodel_contract(self):
        """ fit(X, y) / predict(X) work end-to-end with zero-init state. """
        model = _make_rnn()
        out = model.fit(X_np, y_np, epochs=2)
        assert out is model  # fit returns self for chaining
        Y = model.predict(X_np)
        # single-arg predict returns only the prediction tensor
        assert isinstance(Y, torch.Tensor)
        assert Y.shape == (T, N_OUT)
        assert not Y.requires_grad

    def test_fit_matches_explicit_zero_state(self):
        """ fit(X, y) is equivalent to threading an explicit zero state. """
        torch.manual_seed(0)
        model_a = _make_rnn()
        torch.manual_seed(0)
        model_b = _make_rnn()
        # path A: SignalModel fit
        model_a.fit(X_t, y_t, epochs=3)
        # path B: explicit zero-state threading
        H = torch.zeros(T, model_b.H)
        for _ in range(3):
            _, H = model_b.train_on(X_t, y_t, H)
        with torch.no_grad():
            Y_a = model_a.predict(X_t)
            Y_b, _ = model_b.predict(X_t, torch.zeros(T, model_b.H))
        assert torch.allclose(Y_a, Y_b, atol=1e-6)

    def test_predict_explicit_state_still_returns_tuple(self):
        """ predict(X, H) keeps the explicit-state (Y, H) contract. """
        model = _make_rnn()
        H = torch.zeros(T, model.H)
        Y, H_out = model.predict(X_t, H)
        assert Y.shape == (T, N_OUT)
        assert H_out.shape == (T, model.H)

    def test_predict_moves_input_to_model_device(self):
        """ predict coerces / moves X to the model's parameter device. """
        model = _make_rnn()
        device = next(model.parameters()).device
        # numpy input (not yet a tensor, not on device) must still work
        Y = model.predict(X_np)
        assert Y.device == device
