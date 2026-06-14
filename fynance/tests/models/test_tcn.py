#!/usr/bin/env python3
# coding: utf-8

""" Tests for the Temporal Convolutional Network model. """

# Third-party packages
import pytest
import torch
import torch.nn as nn

# Local packages
from fynance.models.loss import SharpeLoss
from fynance.models.tcn import TemporalConvNet

T, N_IN, N_OUT = 60, 3, 1


@pytest.fixture
def data():
    torch.manual_seed(0)
    X = torch.randn(T, N_IN)
    y = torch.randn(T, N_OUT)
    return X, y


def test_forward_shape(data):
    X, y = data
    model = TemporalConvNet(X, y, channels=[8, 8], kernel_size=2)
    out = model(X)
    assert out.shape == (T, N_OUT)


def test_construct_from_dims():
    model = TemporalConvNet(N_IN, N_OUT, channels=[4])
    assert model.N == N_IN and model.M == N_OUT
    out = model(torch.randn(20, N_IN))
    assert out.shape == (20, N_OUT)


def test_dilation_doubles_with_depth(data):
    X, y = data
    model = TemporalConvNet(X, y, channels=[4, 4, 4], kernel_size=2)
    dilations = [blk.net[0].dilation[0] for blk in model.tcn]
    assert dilations == [1, 2, 4]


def test_gradient_flows(data):
    X, y = data
    model = TemporalConvNet(X, y, channels=[8, 8])
    out = model(X)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_no_lookahead_strictly_causal(data):
    # Output[:t] must not change when the future X[t:] is perturbed.
    X, y = data
    model = TemporalConvNet(X, y, channels=[8, 8], kernel_size=2)
    model.eval()  # disable dropout for determinism
    t = 35
    with torch.no_grad():
        base = model(X)
        X_future = X.clone()
        X_future[t:] += 100.0
        perturbed = model(X_future)
    assert torch.allclose(base[:t], perturbed[:t], atol=1e-5)


def test_train_step_with_mse(data):
    X, y = data
    model = TemporalConvNet(X, y, channels=[8, 8])
    model.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=1e-2)
    loss = model.train_on(model.X, model.y)
    assert torch.isfinite(loss)


def test_train_step_with_sharpe_loss(data):
    X, y = data
    model = TemporalConvNet(X, y, channels=[8, 8])
    model.set_optimizer(SharpeLoss, torch.optim.Adam, lr=1e-2)
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        model.train_on(model.X, model.y)
    after = list(model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_predict_detached(data):
    X, y = data
    model = TemporalConvNet(X, y, channels=[8])
    pred = model.predict(model.X)
    assert pred.shape == (T, N_OUT)
    assert not pred.requires_grad
