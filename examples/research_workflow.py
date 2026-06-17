#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" The canonical fynance research workflow, end-to-end, on synthetic data.

This is the runnable, portable form of the strategy-research loop: synthetic data
→ causal feature matrix (``X``/``y``) → a rule-based **and** an objective-aligned
strategy → a seeded, cost-aware, walk-forward backtest via
:func:`fynance.research.run_experiment` → the guardrails (permutation p-value +
deflated Sharpe) → a portable report.

It is **data-agnostic**: fynance ships only synthetic generators (``gbm`` /
``regime_switching``); point ``X``/``data`` at your own ``PriceSeries`` to run it
on real data. fynance never stores results — every artifact is written under the
``output_dir`` you pass.

Run it::

    python examples/research_workflow.py            # writes to ./research_out

Synthetic results are a **plumbing check, not evidence of edge**: a real claim
needs real data *and* a small permutation p-value *and* a healthy deflated Sharpe.
"""

# Built-in
from __future__ import annotations

from pathlib import Path

# Third-party
import numpy as np

# Local
import fynance as fy
from fynance.backtest import ProportionalCost
from fynance.models import ObjectiveModel, SharpeLoss
from fynance.research import (
    Ledger,
    gbm,
    permutation_test,
    run_experiment,
    write_report,
)
from fynance.strategy import Strategy

FEE = 0.001  # ~10 bps round-trip proportional cost


def build_features(prices, *, train: int):
    """ Build a small **causal** feature matrix ``X`` and target ``y``.

    Columns are trailing/rolling only (no lookahead) and standardized with
    **train-only** statistics. ``y`` is the next-bar return. Returns
    ``(X, y, names)`` ready for ``run_experiment(..., X=X, y=y)``.
    """
    p = prices.to_numpy() if hasattr(prices, "to_numpy") else np.asarray(prices)
    p = p.astype(np.float64)

    cols = [fy.roc(p, 1), fy.roc(p, 5), fy.realized_volatility(p, w=20)]
    names = ["roc_1", "roc_5", "realized_vol_20"]
    X = np.column_stack(cols)

    mu = X[:train].mean(axis=0)
    sd = X[:train].std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y = np.zeros((p.shape[0], 1), dtype=np.float32)
    y[:-1, 0] = (p[1:] / p[:-1] - 1.0).astype(np.float32)

    return X, y, names


def main(output_dir: str | Path = "research_out") -> dict:
    """ Run the full workflow and return the two experiments' metrics. """
    output_dir = Path(output_dir)

    # 1. Data — synthetic here; swap for your own PriceSeries on real data.
    prices = gbm(1500, seed=7)  # a PriceSeries named "synthetic-gbm"
    train, test = 750, 250

    # 2. Features — causal X/y (the rolling-NN refit slices these per window).
    X, y, names = build_features(prices, train=train)

    ledger = Ledger(output_dir / "ledger")

    # 3a. A rule-based baseline (price-only) — used for the permutation test.
    baseline = Strategy(features=lambda p: fy.roc(p, 20), signal=np.sign,
                        cost=ProportionalCost(FEE))
    base_exp = run_experiment(
        baseline, prices, name="ts_momentum",
        walk_forward={"train": train, "test": test}, seed=0,
        data_desc="synthetic GBM (plumbing check, not real data)",
        output_dir=output_dir,
    )

    # 3b. An objective-aligned NN: the net outputs positions, trained directly to
    #     maximize the Sharpe of positions * returns (not MSE on a target).
    model = ObjectiveModel(layers=(16, 8), loss=SharpeLoss(), epochs=60, seed=0)
    objective = Strategy(model=model, signal=lambda pos: pos,
                         cost=ProportionalCost(FEE))
    obj_exp = run_experiment(
        objective, prices, name="sharpe_nn", X=X, y=y,
        walk_forward={"train": train, "test": test}, seed=0,
        feature_names=names, feature_desc="3 causal cols, train-standardized",
        data_desc="synthetic GBM (plumbing check, not real data)",
        output_dir=output_dir,
    )

    # 4. Guardrails — never read the raw Sharpe alone.
    perm = permutation_test(baseline, prices, n_permutations=100, seed=0)
    for exp in (base_exp, obj_exp):
        ledger.append(exp)

    # 5. Reports — portable artifacts under output_dir (viewable on GitHub).
    for exp in (base_exp, obj_exp):
        write_report(exp, output_dir, notebook=False)

    print(f"baseline  sharpe={base_exp.metrics['sharpe']:.3f}  "
          f"perm p={perm['p_value']:.3f}  "
          f"DSR={ledger.deflated_sharpe(base_exp):.3f}")
    print(f"objective sharpe={obj_exp.metrics['sharpe']:.3f}  "
          f"DSR={ledger.deflated_sharpe(obj_exp):.3f}")
    print(f"artifacts under {output_dir.resolve()}  "
          f"(synthetic = plumbing check, not edge)")

    return {"baseline": base_exp.metrics, "objective": obj_exp.metrics,
            "permutation": perm}


if __name__ == "__main__":
    main()
