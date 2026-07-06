#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Objective-aligned training: ObjectiveModel maximizes a financial loss. """

# Third-party
import numpy as np
import pytest
import torch

# Local
from fynance.core import SignalModel
from fynance.metrics import sharpe
from fynance.models import ObjectiveModel, SortinoLoss, pretrain_pooled


def _edge_data(n=1500, seed=0):
    """ A learnable edge: feature s in {-1,1} predicts the next return's sign. """
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=n)
    noise = rng.standard_normal(n) * 0.01
    returns = (s * 0.01 + noise).astype(np.float32)  # position s earns ~+1%/bar
    X = np.column_stack([s, rng.standard_normal(n)]).astype(np.float32)
    return X, returns


def test_conforms_to_signalmodel():
    assert isinstance(ObjectiveModel(), SignalModel)


def test_learns_a_known_edge():
    X, returns = _edge_data()
    model = ObjectiveModel(layers=(8,), epochs=150, lr=5e-3, seed=0).fit(X, returns)

    pos = np.asarray(model.predict(X)).reshape(-1)
    strat_ret = pos * returns
    # The learned positions should align with the edge -> clearly positive Sharpe.
    assert sharpe(np.cumprod(1 + strat_ret), period=252) > 1.0
    # and beat the do-nothing / wrong-way baseline.
    assert strat_ret.mean() > 0.0


def test_positions_are_bounded():
    X, returns = _edge_data(n=300)
    pos = np.asarray(ObjectiveModel(epochs=10).fit(X, returns).predict(X))
    assert pos.shape == (300, 1)
    assert np.all(np.abs(pos) <= 1.0 + 1e-6)


def test_reproducible_with_seed():
    X, returns = _edge_data(n=400)
    a = ObjectiveModel(epochs=20, seed=7).fit(X, returns).predict(X)
    b = ObjectiveModel(epochs=20, seed=7).fit(X, returns).predict(X)
    assert np.allclose(a, b)


def test_warm_started_refit_is_reproducible():
    # The net warm-starts across successive fit() calls (walk-forward refit).
    # Two equal-seed refit sequences over the same data must yield identical
    # predictions -- no hidden non-determinism in the online adaptation.
    X1, r1 = _edge_data(n=300, seed=1)
    X2, r2 = _edge_data(n=300, seed=2)

    def refit_sequence():
        m = ObjectiveModel(layers=(8,), epochs=15, lr=5e-3, seed=11)
        m.fit(X1, r1)        # first walk-forward window
        m.fit(X2, r2)        # warm-started refit on the next window
        return np.asarray(m.predict(X2))

    a = refit_sequence()
    b = refit_sequence()
    assert np.allclose(a, b)


def test_accepts_custom_net_and_loss():
    X, returns = _edge_data(n=300)
    net = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(),
                              torch.nn.Linear(4, 1))
    model = ObjectiveModel(net=net, loss=SortinoLoss(), epochs=10).fit(X, returns)
    assert np.asarray(model.predict(X)).shape == (300, 1)


def _panel_edge(n=1500, n_assets=3, m_features=2, seed=0):
    """ A learnable panel edge: each asset's own feature predicts its return.

    Returns ``X`` of shape ``(n, n_assets, m_features)`` (the first feature of
    each asset is its sign signal, the rest are noise) and ``y`` of shape
    ``(n, n_assets)`` (each asset's realized per-bar return).
    """
    rng = np.random.default_rng(seed)
    feats, rets = [], []
    for _ in range(n_assets):
        s = rng.choice([-1.0, 1.0], size=n)
        r = (s * 0.01 + rng.standard_normal(n) * 0.01).astype(np.float32)
        cols = [s] + [rng.standard_normal(n) for _ in range(m_features - 1)]
        feats.append(np.column_stack(cols).astype(np.float32))
        rets.append(r)
    X = np.stack(feats, axis=1)                     # (n, n_assets, m_features)
    y = np.column_stack(rets).astype(np.float32)    # (n, n_assets)

    return X, y


def _book_turnover(pos):
    """ Mean per-bar turnover aggregated across the book's asset columns. """
    pos = np.asarray(pos)
    return float(np.abs(np.diff(pos, axis=0, prepend=0.0)).sum(axis=1).mean())


def test_panel_predict_shape_3d_and_flat():
    # A 3-D panel (T, N, M) and its pre-flattened (T, N*M) form must both be
    # accepted and yield the *same* position book of shape (T, N).
    X, y = _panel_edge(n=300, n_assets=3, m_features=2)
    model = ObjectiveModel(layers=(8,), epochs=20, lr=5e-3, seed=0).fit(X, y)
    pos = np.asarray(model.predict(X))
    assert pos.shape == (300, 3)
    assert model.n_assets == 3

    Xflat = X.reshape(X.shape[0], -1)
    model2 = ObjectiveModel(n_assets=3, layers=(8,), epochs=20, lr=5e-3,
                            seed=0).fit(Xflat, y)
    pos2 = np.asarray(model2.predict(Xflat))
    assert pos2.shape == (300, 3)
    assert np.allclose(pos, pos2)


def test_panel_positions_are_bounded():
    # With tanh, each per-asset position stays within [-1, 1].
    X, y = _panel_edge(n=300, n_assets=3)
    pos = np.asarray(ObjectiveModel(epochs=10, seed=0).fit(X, y).predict(X))
    assert pos.shape == (300, 3)
    assert np.all(np.abs(pos) <= 1.0 + 1e-6)


def test_panel_book_learns_a_known_edge():
    # Each of the N=3 assets has its own learnable edge; the aggregated book
    # return should achieve a clearly positive Sharpe.
    X, y = _panel_edge(n_assets=3, seed=0)
    model = ObjectiveModel(layers=(8,), epochs=150, lr=5e-3, seed=0).fit(X, y)
    pos = np.asarray(model.predict(X))
    assert pos.shape == (1500, 3)

    book_ret = (pos * y).sum(axis=1)
    assert sharpe(np.cumprod(1 + book_ret), period=252) > 1.0


def test_panel_cost_reduces_book_turnover():
    # On a fast-flipping panel, the turnover-penalized book churns less than the
    # cost-free one (aggregated across asset columns).
    s = np.where(np.arange(1500) % 2 == 0, 1.0, -1.0)
    rng = np.random.default_rng(0)
    feats, rets = [], []
    for _ in range(3):
        r = (s * 0.01 + rng.standard_normal(1500) * 0.005).astype(np.float32)
        feats.append(np.column_stack([s, rng.standard_normal(1500)]
                                     ).astype(np.float32))
        rets.append(r)
    X = np.stack(feats, axis=1)
    y = np.column_stack(rets).astype(np.float32)

    free = ObjectiveModel(layers=(8,), epochs=200, lr=5e-3, seed=0).fit(X, y)
    pricey = ObjectiveModel(layers=(8,), epochs=200, lr=5e-3, cost=0.1,
                            seed=0).fit(X, y)

    assert _book_turnover(pricey.predict(X)) < _book_turnover(free.predict(X))


def test_n1_explicit_matches_inferred():
    # Strict N=1 non-regression: an explicit n_assets=1 run, the inferred path
    # from a 1-D y, and a 2-D (T, 1) y must all give identical positions, with a
    # (T, 1) shape -- the single-asset behaviour is unchanged.
    X, returns = _edge_data(n=400)
    inferred = ObjectiveModel(epochs=20, seed=42).fit(X, returns).predict(X)
    explicit = ObjectiveModel(n_assets=1, epochs=20,
                              seed=42).fit(X, returns).predict(X)
    twod_y = ObjectiveModel(epochs=20, seed=42).fit(
        X, returns.reshape(-1, 1)).predict(X)

    assert inferred.shape == (400, 1)
    assert np.array_equal(inferred, explicit)
    assert np.array_equal(inferred, twod_y)


def _alternating_edge(n=1500, seed=0):
    """ Sign flips every bar: the no-cost optimum churns (turnover ~2/bar). """
    s = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    rng = np.random.default_rng(seed)
    returns = (s * 0.01 + rng.standard_normal(n) * 0.005).astype(np.float32)
    X = np.column_stack([s, rng.standard_normal(n)]).astype(np.float32)

    return X, returns


def _turnover(pos):
    return float(np.abs(np.diff(np.asarray(pos).reshape(-1), prepend=0.0)).mean())


def test_cost_reduces_turnover():
    # On a fast-flipping edge, a no-cost model churns; a turnover-penalized one
    # holds. The cost term should cut realized turnover substantially.
    X, returns = _alternating_edge()
    free = ObjectiveModel(layers=(8,), epochs=200, lr=5e-3, seed=0).fit(X, returns)
    pricey = ObjectiveModel(layers=(8,), epochs=200, lr=5e-3, cost=0.1,
                            seed=0).fit(X, returns)

    assert _turnover(pricey.predict(X)) < _turnover(free.predict(X))


def test_cost_default_zero_is_unchanged():
    # cost defaults to 0 -> identical to not passing it (pure refactor safety).
    X, returns = _edge_data(n=400)
    a = ObjectiveModel(epochs=20, seed=3).fit(X, returns).predict(X)
    b = ObjectiveModel(epochs=20, cost=0.0, seed=3).fit(X, returns).predict(X)
    assert np.allclose(a, b)


def _sr(model, X, returns):
    pos = np.asarray(model.predict(X)).reshape(-1)
    return sharpe(np.cumprod(1 + pos * returns), period=252)


def test_minibatch_trains_more_than_full_batch():
    # Full-batch does only `epochs` gradient steps; mini-batching does
    # `epochs * n_chunks` -> at the same (low) epoch budget it should learn more.
    X, returns = _edge_data()
    full = ObjectiveModel(layers=(8,), epochs=5, lr=5e-3, seed=0).fit(X, returns)
    mini = ObjectiveModel(layers=(8,), epochs=5, lr=5e-3, batch_size=256,
                          seed=0).fit(X, returns)

    assert _sr(mini, X, returns) > _sr(full, X, returns)


def test_minibatch_reproducible_with_seed():
    X, returns = _edge_data(n=600)
    a = ObjectiveModel(epochs=8, batch_size=128, seed=5).fit(X, returns).predict(X)
    b = ObjectiveModel(epochs=8, batch_size=128, seed=5).fit(X, returns).predict(X)
    assert np.allclose(a, b)


def test_minibatch_with_cost_reduces_turnover():
    X, returns = _alternating_edge()
    free = ObjectiveModel(layers=(8,), epochs=20, lr=5e-3, batch_size=256,
                          seed=0).fit(X, returns)
    pricey = ObjectiveModel(layers=(8,), epochs=20, lr=5e-3, batch_size=256,
                            cost=0.1, seed=0).fit(X, returns)

    assert _turnover(pricey.predict(X)) < _turnover(free.predict(X))


# -- Persistence: save / load ------------------------------------------------


def test_save_load_roundtrip_and_trainable(tmp_path):
    # save -> load reconstructs bit-identical predictions and stays trainable.
    X, returns = _edge_data(n=400)
    m = ObjectiveModel(layers=(8,), epochs=20, seed=3).fit(X, returns)

    path = tmp_path / "objective.pt"
    m.save(path)
    loaded = ObjectiveModel.load(path)

    assert np.allclose(m.predict(X), loaded.predict(X), atol=1e-7)

    # Continued trainability: a fit() after load runs and moves the weights.
    before = np.asarray(loaded.predict(X)).copy()
    loaded.fit(X, returns)
    after = np.asarray(loaded.predict(X))
    assert after.shape == before.shape
    assert not np.array_equal(before, after)


def test_save_load_preserves_config(tmp_path):
    # The reloaded model carries the same init config (needed to reconstruct).
    m = ObjectiveModel(layers=(8, 4), n_assets=1, epochs=13, lr=2e-3,
                       batch_size=64, shuffle=False, cost=0.01, seed=9)
    X, returns = _edge_data(n=200)
    m.fit(X, returns)

    path = tmp_path / "objective.pt"
    m.save(path)
    loaded = ObjectiveModel.load(path)

    for attr in ("n_assets", "layers", "epochs", "lr", "batch_size", "shuffle",
                 "cost", "seed"):
        assert getattr(loaded, attr) == getattr(m, attr)


# -- clone -------------------------------------------------------------------


def test_clone_same_weights_independent_training():
    X, returns = _edge_data(n=400)
    m = ObjectiveModel(layers=(8,), epochs=20, seed=4).fit(X, returns)

    clone = m.clone()
    # Same (deep-copied) weights -> identical predictions.
    assert np.allclose(m.predict(X), clone.predict(X), atol=1e-7)

    # Training the clone leaves the original bit-identical, and the clone moves.
    original_before = np.asarray(m.predict(X)).copy()
    clone.fit(X, returns)
    assert np.array_equal(original_before, np.asarray(m.predict(X)))
    assert not np.array_equal(original_before, np.asarray(clone.predict(X)))


# -- finetune ----------------------------------------------------------------


def _trunk_head_names(model):
    """ Split the net's state_dict keys into (trunk, head) parameter names. """
    head = model._head_module()
    head_names = set()
    for name, mod in model.net.named_modules():
        if mod is head:
            for pn, _ in mod.named_parameters(recurse=False):
                head_names.add(f"{name}.{pn}" if name else pn)
    all_names = set(model.net.state_dict().keys())

    return all_names - head_names, head_names


def test_finetune_freeze_trunk_trains_head_only():
    X, returns = _edge_data(n=400)
    m = ObjectiveModel(layers=(8,), epochs=30, seed=5).fit(X, returns)

    trunk_names, head_names = _trunk_head_names(m)
    assert trunk_names and head_names  # the default MLP has both a trunk & head

    before = {k: v.clone() for k, v in m.net.state_dict().items()}
    m.finetune(X, returns, freeze_trunk=True, epochs=30)
    after = m.net.state_dict()

    # Trunk parameters are byte-identical; head parameters changed.
    assert all(torch.equal(before[k], after[k]) for k in trunk_names)
    assert any(not torch.equal(before[k], after[k]) for k in head_names)


def test_finetune_no_freeze_changes_trunk():
    X, returns = _edge_data(n=400)
    m = ObjectiveModel(layers=(8,), epochs=30, seed=5).fit(X, returns)

    trunk_names, _ = _trunk_head_names(m)
    before = {k: v.clone() for k, v in m.net.state_dict().items()}
    m.finetune(X, returns, freeze_trunk=False, epochs=30)
    after = m.net.state_dict()

    assert any(not torch.equal(before[k], after[k]) for k in trunk_names)


def test_finetune_before_fit_raises():
    X, returns = _edge_data(n=100)
    with pytest.raises(RuntimeError):
        ObjectiveModel().finetune(X, returns)


# -- pretrain_pooled ---------------------------------------------------------


def test_pretrain_pooled_batches_never_cross_joins():
    # Segment lengths that batch_size does NOT divide: a naive global-contiguous
    # batching over the concatenation WOULD span a join -> the check has teeth.
    lengths = [1000, 800]
    bs = 256
    assert any(L % bs != 0 for L in lengths)

    Xs, ys = [], []
    for i, L in enumerate(lengths):
        X, r = _edge_data(n=L, seed=i)
        Xs.append(X)
        ys.append(r)

    model = ObjectiveModel(layers=(8,), epochs=2, batch_size=bs, seed=0)
    pretrain_pooled(model, Xs, ys)

    offsets = np.cumsum([0] + lengths)
    for si, a, b in model._batch_plan:
        assert 0 <= a < b <= lengths[si]                 # slice within a segment
        g0, g1 = offsets[si] + a, offsets[si] + b        # global index range ...
        assert offsets[si] <= g0 < g1 <= offsets[si + 1]  # ... inside one asset
    # Every asset contributed at least one batch.
    assert {si for si, _, _ in model._batch_plan} == set(range(len(lengths)))


def _shared_signal_asset(n, seed, beta=0.01, noise=0.02):
    """ An asset whose first feature is a planted linear driver of its return. """
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n).astype(np.float32)          # the shared signal
    nf = rng.standard_normal(n).astype(np.float32)         # a noise feature
    r = (beta * s + rng.standard_normal(n) * noise).astype(np.float32)
    X = np.column_stack([s, nf]).astype(np.float32)

    return X, r


def _heldout_sharpe(model, X, r):
    pos = np.asarray(model.predict(X)).reshape(-1)

    return sharpe(np.cumprod(1 + pos * r), period=252)


def test_pretrain_pooled_beats_single_asset():
    # Two assets share a planted linear signal (different noise). Pooling
    # asset-0's train slice with asset-1's should beat training on asset-0's
    # slice alone, judged on asset-0's held-out slice. Averaged over 3 seeds to
    # stay robust to the idiosyncratic noise.
    n = 150
    singles, pooleds = [], []
    for seed in range(3):
        X0, r0 = _shared_signal_asset(2 * n, seed)
        X0tr, r0tr, X0te, r0te = X0[:n], r0[:n], X0[n:], r0[n:]
        X1, r1 = _shared_signal_asset(n, seed + 1000)

        single = ObjectiveModel(layers=(8,), epochs=60, lr=5e-3,
                                seed=seed).fit(X0tr, r0tr)
        pooled = ObjectiveModel(layers=(8,), epochs=60, lr=5e-3, seed=seed)
        pretrain_pooled(pooled, [X0tr, X1], [r0tr, r1])

        singles.append(_heldout_sharpe(single, X0te, r0te))
        pooleds.append(_heldout_sharpe(pooled, X0te, r0te))

    assert np.mean(pooleds) > np.mean(singles)


def test_pretrain_pooled_mismatched_lengths_raises():
    X, r = _edge_data(n=100)
    with pytest.raises(ValueError):
        pretrain_pooled(ObjectiveModel(epochs=2), [X, X], [r])
    with pytest.raises(ValueError):
        pretrain_pooled(ObjectiveModel(epochs=2), [], [])


def test_pretrain_pooled_mismatched_features_raises():
    X, r = _edge_data(n=100)
    X_wide = np.column_stack([X, X]).astype(np.float32)  # 4 features vs 2
    with pytest.raises(ValueError):
        pretrain_pooled(ObjectiveModel(epochs=2), [X, X_wide], [r, r])
