#!/usr/bin/env python3
# coding: utf-8

""" Tests for the causal Transformer model. """

# Third-party packages
import pytest
import torch
import torch.nn as nn

# Local packages
from fynance.models.loss import SharpeLoss
from fynance.models.transformer import PositionalEncoding, Transformer

T, N_IN, N_OUT = 40, 3, 1


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(T, N_IN), torch.randn(T, N_OUT)


def _make(X, y, **kw):
    kw.setdefault("d_model", 16)
    kw.setdefault("num_heads", 2)
    kw.setdefault("num_layers", 2)
    return Transformer(X, y, **kw)


def test_forward_shape(data):
    X, y = data
    assert _make(X, y)(X).shape == (T, N_OUT)


def test_construct_from_dims():
    model = _make(N_IN, N_OUT)
    assert model.N == N_IN and model.M == N_OUT
    assert model(torch.randn(15, N_IN)).shape == (15, N_OUT)


def test_num_layers(data):
    X, y = data
    model = _make(X, y, num_layers=3)
    assert len(model.blocks) == 3


def test_d_model_not_divisible_raises(data):
    X, y = data
    with pytest.raises(ValueError, match="divisible"):
        _make(X, y, d_model=15, num_heads=2)


def test_positional_encoding_adds_signal():
    pe = PositionalEncoding(8)
    x = torch.zeros(1, 5, 8)
    out = pe(x)
    assert out.shape == (1, 5, 8)
    assert torch.any(out != 0)  # encoding added to zeros


def test_no_lookahead_strictly_causal(data):
    # The causal mask must prevent any position from seeing the future.
    X, y = data
    model = _make(X, y)
    model.eval()
    t = 25
    with torch.no_grad():
        base = model(X)
        X_future = X.clone()
        X_future[t:] += 100.0
        perturbed = model(X_future)
    assert torch.allclose(base[:t], perturbed[:t], atol=1e-5)


def test_gradient_flows(data):
    X, y = data
    model = _make(X, y)
    loss = ((model(X) - y) ** 2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_train_step_with_mse(data):
    X, y = data
    model = _make(X, y)
    model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-3)
    assert torch.isfinite(model.train_on(model.X, model.y))


def test_train_step_with_sharpe_loss(data):
    X, y = data
    model = _make(X, y)
    model.set_optimizer(SharpeLoss, torch.optim.Adam, lr=1e-2)
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        model.train_on(model.X, model.y)
    after = list(model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_predict_detached(data):
    X, y = data
    model = _make(X, y)
    pred = model.predict(model.X)
    assert pred.shape == (T, N_OUT) and not pred.requires_grad
