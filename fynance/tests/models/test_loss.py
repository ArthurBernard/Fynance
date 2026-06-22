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

    def test_rf_change_takes_effect(self):
        # _rf_per_period is recomputed in forward (a property), so mutating rf
        # after construction must change the loss instead of being a no-op.
        loss_fn = SharpeLoss()
        before = loss_fn(RETURNS).item()
        loss_fn.rf = 0.10
        after = loss_fn(RETURNS).item()
        assert after != before
        # Matches building a fresh loss with the same rf.
        assert after == pytest.approx(SharpeLoss(rf=0.10)(RETURNS).item())


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

    def test_all_positive_finite_bounded_and_scale_invariant(self):
        # On an all-gains batch the downside is ~0. A *fixed absolute* eps inside
        # the sqrt is dimensionally wrong: the loss then scales with the return
        # magnitude (10x returns -> 10x loss) instead of being scale-invariant
        # like a true Sortino ratio, and explodes. A returns-scaled floor keeps
        # the loss finite, bounded, and scale-invariant.
        r = torch.abs(RETURNS) + 0.001   # strictly positive → downside = 0
        loss = SortinoLoss(eps=1e-8)(r)
        loss_10x = SortinoLoss(eps=1e-8)(10.0 * r)
        assert torch.isfinite(loss)
        assert abs(loss.item()) <= 1e3 + 1.0          # bounded
        # scale invariance (old absolute-eps code is off by ~10x here)
        assert loss_10x.item() == pytest.approx(loss.item(), rel=1e-2)

    def test_low_risk_gradient_survives(self):
        # On a strong-uptrend (near-zero-downside) batch the ratio blows past
        # MAX_RATIO. A hard clamp pinned the loss to a constant there and so
        # ZEROED the gradient in exactly the regime we still want to optimize.
        # The smooth tanh saturation must keep the loss finite AND leave a
        # non-zero gradient w.r.t. the input.
        gen = torch.Generator().manual_seed(0)
        r = (torch.rand(60, generator=gen) * 0.01 + 0.005).requires_grad_(True)
        loss = SortinoLoss()(r)
        assert torch.isfinite(loss)
        loss.backward()
        assert r.grad is not None
        assert torch.any(r.grad != 0)
        assert not torch.isnan(r.grad).any()

    def test_higher_sortino_gives_lower_loss(self):
        # Sign convention must survive the smooth saturation: a higher-Sortino
        # batch (more upside, same downside) must give a strictly lower loss
        # even when both batches are well into the saturating regime.
        gen = torch.Generator().manual_seed(1)
        base = torch.rand(60, generator=gen) * 0.01 + 0.005
        better = base + 0.01   # uniformly higher mean, downside still ~0
        assert SortinoLoss()(better).item() < SortinoLoss()(base).item()

    def test_positive_when_negative_mean(self):
        # Sign convention (like SharpeLoss): a negative-mean return series has a
        # negative Sortino ratio, so the negated loss must be positive.
        loss = SortinoLoss()(RETURNS_NEG)
        assert loss.item() > 0

    def test_rf_change_takes_effect(self):
        # _rf_per_period is a property (not cached in __init__), so mutating rf
        # after construction must change the computed loss.
        loss_fn = SortinoLoss()
        before = loss_fn(RETURNS).item()
        loss_fn.rf = 5.0   # huge rf -> excess turns negative -> loss flips sign
        after = loss_fn(RETURNS).item()
        assert before != after

    def test_downside_only_penalized(self):
        # Two series identical except one has large upside: Sortino should
        # be strictly lower (better) for the one with large upside because
        # upside doesn't count against the denominator.
        r_base = torch.tensor([-0.01, 0.005, -0.008, 0.003, -0.006])
        r_upside = torch.tensor([-0.01, 0.500, -0.008, 0.003, -0.006])
        loss_base = SortinoLoss()(r_base)
        loss_upside = SortinoLoss()(r_upside)
        # Large upside improves mean without worsening downside → lower loss
        assert loss_upside.item() < loss_base.item()


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


# ---------------------------------------------------------------------------
# §5.1 new losses: Calmar, Omega, Hybrid
# ---------------------------------------------------------------------------

class TestCalmarLoss:
    def test_scalar_and_grad(self):
        from fynance.models.loss import CalmarLoss
        r = torch.randn(120, 1, requires_grad=True)
        loss = CalmarLoss()(r)
        assert loss.ndim == 0
        loss.backward()
        assert r.grad is not None and torch.any(r.grad != 0)

    def test_rejects_non_tensor(self):
        from fynance.models.loss import CalmarLoss
        with pytest.raises(TypeError):
            CalmarLoss()(np.zeros(10))

    def test_no_drawdown_loss_is_bounded(self):
        from fynance.models.loss import CalmarLoss
        # Monotonically increasing equity -> zero drawdown. A fixed absolute
        # eps (1e-8) on an O(returns) drawdown made the ratio explode (e.g.
        # -3.78e7). A returns-scaled floor keeps it finite and bounded.
        r = torch.full((100,), 0.01)   # constant gains -> no drawdown
        loss = CalmarLoss(eps=1e-8)(r)
        assert torch.isfinite(loss)
        assert abs(loss.item()) < 1e4

    def test_all_zero_returns_is_finite(self):
        from fynance.models.loss import CalmarLoss
        # Degenerate all-zero series: numerator and drawdown are both 0; the
        # bare-eps backstop in the floor must keep the loss finite (not NaN).
        loss = CalmarLoss(eps=1e-8)(torch.zeros(50))
        assert torch.isfinite(loss)

    def test_low_drawdown_gradient_survives(self):
        from fynance.models.loss import CalmarLoss
        # Near-zero-drawdown (almost monotone equity) batch: the ratio blows
        # past MAX_RATIO. A hard clamp pinned the loss to a constant and ZEROED
        # the gradient there; the smooth tanh saturation must keep the loss
        # finite AND leave a non-zero gradient w.r.t. the input.
        gen = torch.Generator().manual_seed(0)
        r = (torch.rand(100, generator=gen) * 0.01 + 0.005).requires_grad_(True)
        loss = CalmarLoss()(r)
        assert torch.isfinite(loss)
        loss.backward()
        assert r.grad is not None
        assert torch.any(r.grad != 0)
        assert not torch.isnan(r.grad).any()

    def test_higher_calmar_gives_lower_loss(self):
        from fynance.models.loss import CalmarLoss
        # Sign convention must survive the smooth saturation: a higher-Calmar
        # batch (higher return, same near-zero drawdown) gives a lower loss.
        gen = torch.Generator().manual_seed(1)
        base = torch.rand(100, generator=gen) * 0.01 + 0.005
        better = base + 0.01   # higher mean return, drawdown still ~0
        assert CalmarLoss()(better).item() < CalmarLoss()(base).item()


class TestOmegaLoss:
    def test_known_value(self):
        from fynance.models.loss import OmegaLoss
        # gains mean = (1+3)/4 = 1 ; losses mean = (2+0+0+0... ) -> compute
        r = torch.tensor([1.0, -2.0, 3.0, -1.0])
        gains = torch.relu(r).mean()
        losses = torch.relu(-r).mean()
        expected = -(gains / (losses + 1e-8))
        assert torch.isclose(OmegaLoss()(r), expected)

    def test_threshold_and_grad(self):
        from fynance.models.loss import OmegaLoss
        r = torch.randn(100, 1, requires_grad=True)
        loss = OmegaLoss(threshold=0.01)(r)
        loss.backward()
        assert r.grad is not None

    def test_all_gains_loss_is_bounded(self):
        from fynance.models.loss import OmegaLoss
        # All returns above the threshold -> zero losses. A fixed absolute eps
        # made the ratio explode (e.g. -1e6); a returns-scaled floor bounds it.
        r = torch.full((100,), 0.01)   # all gains, no losses below threshold
        loss = OmegaLoss(eps=1e-8)(r)
        assert torch.isfinite(loss)
        assert abs(loss.item()) < 1e4

    def test_all_zero_diff_is_finite(self):
        from fynance.models.loss import OmegaLoss
        # Returns exactly at threshold: gains == losses == 0; the bare-eps
        # backstop must keep the loss finite (not NaN).
        loss = OmegaLoss(threshold=0.0, eps=1e-8)(torch.zeros(50))
        assert torch.isfinite(loss)

    def test_all_gains_gradient_survives(self):
        from fynance.models.loss import OmegaLoss
        # All-gains (zero-loss) batch: the ratio blows past MAX_RATIO. A hard
        # clamp pinned the loss to a constant and ZEROED the gradient there;
        # the smooth tanh saturation must keep the loss finite AND leave a
        # non-zero gradient w.r.t. the input.
        gen = torch.Generator().manual_seed(0)
        r = (torch.rand(100, generator=gen) * 0.01 + 0.005).requires_grad_(True)
        loss = OmegaLoss()(r)
        assert torch.isfinite(loss)
        loss.backward()
        assert r.grad is not None
        assert torch.any(r.grad != 0)
        assert not torch.isnan(r.grad).any()

    def test_higher_omega_gives_lower_loss(self):
        from fynance.models.loss import OmegaLoss
        # Sign convention must survive the smooth saturation: an all-gains batch
        # (Omega -> high) must give a strictly lower loss than a mixed batch
        # with real losses (finite, smaller Omega).
        gen = torch.Generator().manual_seed(2)
        all_gains = torch.rand(100, generator=gen) * 0.01 + 0.005
        mixed = torch.rand(100, generator=gen) * 0.02 - 0.01
        assert OmegaLoss()(all_gains).item() < OmegaLoss()(mixed).item()


class TestHybridLoss:
    def test_weighted_sum(self):
        from fynance.models.loss import HybridLoss, SharpeLoss, SortinoLoss
        r = torch.randn(100, 1)
        a, b = SharpeLoss(), SortinoLoss()
        h = HybridLoss(a, b, alpha=0.3)
        expected = 0.3 * a(r) + 0.7 * b(r)
        assert torch.isclose(h(r), expected)

    def test_forwards_y_true(self):
        from fynance.models.loss import DirectionalAccuracyLoss, HybridLoss, SharpeLoss
        r = torch.randn(100, 1)
        y = torch.randn(100, 1)
        h = HybridLoss(SharpeLoss(), DirectionalAccuracyLoss(), alpha=0.5)
        assert torch.isfinite(h(r, y))

    def test_learnable_alpha_is_parameter(self):
        from fynance.models.loss import HybridLoss, SharpeLoss, SortinoLoss
        h = HybridLoss(SharpeLoss(), SortinoLoss(), alpha=0.5, learnable=True)
        params = list(h.parameters())
        assert len(params) == 1 and params[0].requires_grad
        r = torch.randn(80, 1)
        before = h._alpha_raw.detach().clone()
        opt = torch.optim.SGD(h.parameters(), lr=1.0)
        for _ in range(5):
            opt.zero_grad()
            h(r).backward()
            opt.step()
        assert not torch.allclose(before, h._alpha_raw)


def test_train_model_with_calmar_loss():
    from fynance.models.loss import CalmarLoss
    from fynance.models.mlp import MultiLayerPerceptron
    X = torch.randn(60, 3)
    y = torch.randn(60, 1)
    model = MultiLayerPerceptron(X, y, layers=[8])
    model.set_optimizer(CalmarLoss, torch.optim.Adam, lr=1e-2)
    loss = model.train_on(model.X, model.y)
    assert torch.isfinite(loss)
