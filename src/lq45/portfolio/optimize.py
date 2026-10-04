"""Minimum-variance and mean-variance weight optimization with IDX retail constraints.

Basis (details: docs/DECISIONS.md):
- Long-only and weights summing to one: Markowitz (1952); shorting is
  not allowed in IDX retail trading (E10).
- Maximum weight 35% is an independent decision; tested at 25%, 35%,
  50%, 100% (E2, E17).
- Zero turnover penalty because costs are simulated explicitly (E12).
- Target-return constrained formulation: Chaweewanchon & Chaysiri (2022)
  Section 3.1 Equations (1)-(4) — min w'Σw s.t. w'μ=γ, Σw=1, 0≤w≤maxw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimum_variance_weights(cov: pd.DataFrame, max_weight: float = 0.35) -> pd.Series:
    """Minimum-variance weights: min w'Sw with 0 <= w <= max, sum w = 1.

    If optimization fails, return equal weights so the pipeline does not
    stop; such failures are logged by the caller.
    """
    tickers = list(cov.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("empty ticker list")
    if n == 1:
        return pd.Series([1.0], index=tickers)
    matrix = cov.to_numpy(dtype=float)
    bounds = [(0.0, float(max_weight))] * n
    sum_constraint = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    initial = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: float(w @ matrix @ w),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        return pd.Series(initial, index=tickers)
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return pd.Series(initial, index=tickers)
    return pd.Series(weights / total, index=tickers)


def mean_variance_target_return_weights(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    target_return: float,
    max_weight: float = 0.35,
    seed: int = 0,
) -> pd.Series:
    """Mean-variance weights: min w'Sw s.t. w'μ=γ, Σw=1, 0<=w<=max.

    Formulation of Chaweewanchon & Chaysiri (2022) Section 3.1
    Equations (1)-(4):
    Minimize σ² = ∑ᵢ∑ⱼ wᵢwⱼCᵢⱼ
    subject to ∑ᵢ wᵢEᵢ = γ, ∑ᵢ wᵢ = 1, wᵢ ≥ 0.

    If optimization fails or `target_return` is infeasible, return the
    GMV (minimum-variance) weights as a fallback. A random (Dirichlet)
    starting point uses a fixed `seed` so results are deterministic.
    """
    tickers = list(cov.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("empty ticker list")
    if n == 1:
        return pd.Series([1.0], index=tickers)

    matrix = cov.to_numpy(dtype=float)
    mu = expected_returns.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

    # Feasibility check: is target_return achievable?
    # With 0<=w<=max and sum(w)=1, the return range is:
    # min_w'μ <= γ <= max_w'μ
    mu_min = mu.min() if n > 0 else 0.0
    mu_max = mu.max() if n > 0 else 0.0
    if target_return < mu_min or target_return > mu_max:
        return minimum_variance_weights(cov, max_weight)

    bounds = [(0.0, float(max_weight))] * n
    sum_constraint = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    target_constraint = {"type": "eq", "fun": lambda w: float(w @ mu - target_return)}

    # Try several starting points if the first one fails.
    rng = np.random.default_rng(seed)
    starting_points = [
        np.full(n, 1.0 / n),
        rng.dirichlet(np.ones(n)),
    ]
    if n >= 3:
        starting_points.append(
            np.concatenate(
                [
                    [target_return / mu_max if mu_max > 0 else 1 / n],
                    np.ones(n - 1)
                    * (1 - target_return / mu_max if mu_max > 0 else 1 / n)
                    / (n - 1),
                ]
            )
        )

    for sp in starting_points:
        sp = np.clip(sp, 0, max_weight)
        sp = sp / sp.sum() if sp.sum() > 0 else np.full(n, 1.0 / n)
        result = minimize(
            lambda w: float(w @ matrix @ w),
            sp,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_constraint, target_constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success:
            weights = np.clip(result.x, 0.0, None)
            total = weights.sum()
            if total > 0 and abs(float(weights @ mu - target_return)) < 1e-4:
                return pd.Series(weights / total, index=tickers)

    # Fallback to GMV.
    return minimum_variance_weights(cov, max_weight)
