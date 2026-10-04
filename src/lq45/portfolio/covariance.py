"""Covariance estimators for minimum-variance optimization.

Basis (details: docs/DECISIONS.md):
- Comparison of four estimators rather than assuming a single winner:
  DeMiguel et al. (2009) show that 1/N is hard to beat, so any claim of
  estimator superiority must be tested; Ledoit & Wolf (2004) propose
  shrinkage for high-dimensional covariance matrices.
- Ridge epsilon = 1e-4 stabilizes the diagonal (E11).
- Window L = 120 trading days; sensitivity at 60 and 252 (E17).
- The `gmv` label uses the sample covariance as in Huang et al. (2024),
  who use Global Minimum Variance in the second stage; it yields the
  same result as `sample` under identical constraints and is kept as a
  named comparison baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def returns_matrix(prices: pd.DataFrame, end: str, days: int) -> pd.DataFrame:
    """Daily log returns for the window `(end-days, end]`.

    `prices` is indexed by date with one column per ticker (Adj Close).
    Only data up to `end` is used so the future is never observed. Rows
    with NaN are dropped per column later by the caller.
    """
    window = prices[prices.index <= end].tail(days + 1)
    returns = np.log(window / window.shift(1)).iloc[1:]
    return returns


def estimate_covariance(
    returns: pd.DataFrame, method: str, ridge_epsilon: float = 1e-4
) -> pd.DataFrame:
    """Estimate the daily covariance matrix from returns.

    Methods: `sample`, `ridge_epsilon`, `ledoit_wolf`, `gmv`.
    `gmv` uses the sample covariance (identical to `sample`).
    """
    clean = returns.dropna(axis=1)
    if clean.shape[1] == 0:
        raise ValueError("no complete return columns")
    if method in ("sample", "gmv"):
        return clean.cov()
    if method == "ridge_epsilon":
        cov = clean.cov().to_numpy()
        cov = cov + ridge_epsilon * np.eye(cov.shape[0])
        return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)
    if method == "ledoit_wolf":
        lw = LedoitWolf().fit(clean.to_numpy())
        return pd.DataFrame(lw.covariance_, index=clean.columns, columns=clean.columns)
    raise ValueError(f"unknown covariance method: {method}")
