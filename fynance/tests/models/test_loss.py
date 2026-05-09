#!/usr/bin/env python3
# coding: utf-8

""" Tests for differentiable financial loss functions. """

# Built-in packages

# Third-party packages
import numpy as np
import pytest
import torch

# Local packages
from fynance.models.loss import (
    DirectionalAccuracyLoss,
    SharpeLoss,
    SortinoLoss,
)

RNG = np.random.default_rng(42)
T = 60
RETURNS = torch.from_numpy(RNG.standard_normal(T).astype(np.float32) * 0.01 + 0.001)
RETURNS_NEG = torch.from_numpy(RNG.standard_normal(T).astype(np.float32) * 0.01 - 0.001)
Y_TRUE = torch.from_numpy(RNG.standard_normal(T).astype(np.float32))
Y_PRED = torch.from_numpy(RNG.standard_normal(T).astype(np.float32))


class TestSharpeLoss:
    def test_forward_scalar(self):
        loss = SharpeLoss()(RETURNS)
        assert loss.shape == torch.Size([])

    def test_negative_when_positive_mean(self):
        # Positive mean excess return → Sharpe > 0 → loss < 0
        loss = SharpeLoss()(RETURNS)
        assert loss.item() < 0

    def test_positive_when_negative_mean(self):
        loss = SharpeLoss()(RETURNS_NEG)
        assert loss.item() > 0

    def test_gradient_flow(self):
        r = RETURNS.clone().requires_grad_(True)
        loss = SharpeLoss()(r)
        loss.backward()
        assert r.grad is not None
        assert not torch.isnan(r.grad).any()

    def test_type_error_on_numpy(self):
        arr = np.array([0.01, -0.02, 0.03])
        with pytest.raises(TypeError, match="torch.Tensor"):
            SharpeLoss()(arr)

    def test_rf_shifts_loss(self):
        loss_no_rf = SharpeLoss(rf=0.)(RETURNS)
        loss_high_rf = SharpeLoss(rf=0.10)(RETURNS)
        # High rf reduces excess returns → lower (or equal) Sharpe → higher loss
        assert loss_high_rf.item() >= loss_no_rf.item()

    def test_constant_returns_no_zero_division(self):
        r = torch.full((T,), 0.001)
        loss = SharpeLoss(eps=1e-8)(r)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestSortinoLoss:
    def test_forward_scalar(self):
        loss = SortinoLoss()(RETURNS)
        assert loss.shape == torch.Size([])

    def test_gradient_flow(self):
        r = RETURNS.clone().requires_grad_(True)
        loss = SortinoLoss()(r)
        loss.backward()
        assert r.grad is not None
        assert not torch.isnan(r.grad).any()

    def test_type_error_on_numpy(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            SortinoLoss()(np.array([0.01, -0.02]))

    def test_all_positive_no_zero_division(self):
        r = torch.abs(RETURNS) + 0.001   # strictly positive → downside = 0
        loss = SortinoLoss(eps=1e-8)(r)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_downside_only_penalized(self):
        # Two series identical except one has large upside: Sortino should
        # be lower (better) for the one with large upside because upside
        # doesn't count against the denominator.
        r_base = torch.tensor([-0.01, 0.005, -0.008, 0.003, -0.006])
        r_upside = torch.tensor([-0.01, 0.500, -0.008, 0.003, -0.006])
        loss_base = SortinoLoss()(r_base)
        loss_upside = SortinoLoss()(r_upside)
        # Large upside improves mean without worsening downside → lower loss
        assert loss_upside.item() <= loss_base.item()


class TestDirectionalAccuracyLoss:
    def test_forward_scalar(self):
        loss = DirectionalAccuracyLoss()(Y_PRED, Y_TRUE)
        assert loss.shape == torch.Size([])

    def test_range(self):
        # Sigmoid output in (0,1) → mean in (0,1) → loss in (-1, 0)
        loss = DirectionalAccuracyLoss()(Y_PRED, Y_TRUE)
        assert -1.0 < loss.item() < 0.0

    def test_perfect_direction_lower_loss(self):
        # Same signs → all sigmoid outputs > 0.5 → loss < -0.5
        y = torch.tensor([1., -1., 1., -1., 1., -1.])
        y_correct = y.clone()
        y_wrong = -y.clone()
        loss_correct = DirectionalAccuracyLoss(temperature=10.)(y_correct, y)
        loss_wrong = DirectionalAccuracyLoss(temperature=10.)(y_wrong, y)
        assert loss_correct.item() < loss_wrong.item()

    def test_gradient_flow(self):
        pred = Y_PRED.clone().requires_grad_(True)
        loss = DirectionalAccuracyLoss()(pred, Y_TRUE)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_type_error_y_pred(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            DirectionalAccuracyLoss()(np.array([1., -1.]), Y_TRUE)

    def test_type_error_y_true(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            DirectionalAccuracyLoss()(Y_PRED, np.array([1., -1.]))
