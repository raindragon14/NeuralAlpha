"""Sharpe significance tests and multiple-testing corrections.

Basis (details: docs/keputusan_desain.md):
- Ledoit & Wolf (2008) Sharpe test: robustifies Jobson & Korkie (1981)
  against non-normality and time-series dependence via HAC (E15).
- Romano & Wolf (2005): stepdown for data snooping; used together with
  PBO from Bailey et al. (2016) as an overfitting control (E16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def newey_west_covariance(series: np.ndarray, lags: int | None = None) -> float:
    """HAC long-run variance with Bartlett weights.

    The default lag follows the 4*(n/100)^(2/9) rule from the
    time-series literature.
    """
    n = len(series)
    if n < 2:
        return 0.0
    if lags is None:
        lags = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    centered = series - series.mean()
    cov = float(np.dot(centered, centered) / n)
    for h in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - h / (lags + 1.0)
        cov += 2.0 * weight * float(np.dot(centered[h:], centered[:-h]) / n)
    return max(cov, 0.0)


def newey_west_covariance_matrix(
    series: np.ndarray, lags: int | None = None
) -> np.ndarray:
    """HAC long-run covariance of the sample mean for `series` (T, k).

    Bartlett weights; the default lag follows the 4*(n/100)^(2/9) rule.
    """
    n, k = series.shape
    if n < 2:
        return np.zeros((k, k))
    if lags is None:
        lags = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    centered = series - series.mean(axis=0)
    cov = centered.T @ centered / n
    for h in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - h / (lags + 1.0)
        gamma_h = centered[h:].T @ centered[:-h] / n
        cov = cov + weight * (gamma_h + gamma_h.T)
    return cov / n


def sharpe_difference_test_lw(
    returns_a: pd.Series, returns_b: pd.Series, periods_per_year: int = 252
) -> dict[str, float]:
    """Ledoit & Wolf (2008) test of H0: SR_a = SR_b, robust to non-normality
    and time-series dependence.

    Delta method over the moment vector (mu_a, mu_b, E[r_a^2], E[r_b^2],
    E[r_a r_b]) with a Newey-West covariance; the statistic is the Sharpe
    difference over its asymptotic standard error.
    """
    merged = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    if len(merged) < 30:
        return {"delta_sharpe": 0.0, "t_stat": 0.0, "p_value": 1.0}
    a = merged.iloc[:, 0].to_numpy(dtype=float)
    b = merged.iloc[:, 1].to_numpy(dtype=float)
    moments = np.column_stack([a, b, a * a, b * b, a * b])
    mu_a, mu_b, m2a, m2b, _ = moments.mean(axis=0)
    var_a = m2a - mu_a * mu_a
    var_b = m2b - mu_b * mu_b
    if var_a <= 0 or var_b <= 0:
        return {"delta_sharpe": 0.0, "t_stat": 0.0, "p_value": 1.0}
    sigma_a = float(np.sqrt(var_a))
    sigma_b = float(np.sqrt(var_b))
    sr_a = float(mu_a / sigma_a)
    sr_b = float(mu_b / sigma_b)
    gradient = np.array(
        [
            1.0 / sigma_a + mu_a * mu_a / sigma_a**3,
            -(1.0 / sigma_b + mu_b * mu_b / sigma_b**3),
            -mu_a / (2.0 * sigma_a**3),
            mu_b / (2.0 * sigma_b**3),
            0.0,
        ]
    )
    omega = newey_west_covariance_matrix(moments)
    variance = float(gradient @ omega @ gradient)
    delta = float((sr_a - sr_b) * np.sqrt(periods_per_year))
    if variance <= 0:
        return {"delta_sharpe": delta, "t_stat": 0.0, "p_value": 1.0}
    t_stat = float((sr_a - sr_b) / np.sqrt(variance))
    p_value = float(2.0 * norm.sf(abs(t_stat)))
    return {"delta_sharpe": delta, "t_stat": t_stat, "p_value": p_value}


def mean_difference_test(
    returns_a: pd.Series, returns_b: pd.Series, periods_per_year: int = 252
) -> dict[str, float]:
    """HAC t-test of H0: mean active return = 0 (supplementary to LW 2008).

    Returns the annualized active-return Sharpe, the Newey-West
    t-statistic on the mean of the difference series, and the two-sided
    p-value.
    """
    merged = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    if len(merged) < 30:
        return {"delta_sharpe": 0.0, "t_stat": 0.0, "p_value": 1.0}
    diff = merged.iloc[:, 0].to_numpy(dtype=float) - merged.iloc[:, 1].to_numpy(
        dtype=float
    )
    vol = float(np.std(diff, ddof=1))
    if vol <= 0:
        return {"delta_sharpe": 0.0, "t_stat": 0.0, "p_value": 1.0}
    delta = float(np.sqrt(periods_per_year) * diff.mean() / vol)
    error = float(np.sqrt(newey_west_covariance(diff) / len(diff)))
    if error <= 0:
        return {"delta_sharpe": delta, "t_stat": 0.0, "p_value": 1.0}
    t_stat = float(diff.mean() / error)
    p_value = float(2.0 * norm.sf(abs(t_stat)))
    return {"delta_sharpe": delta, "t_stat": t_stat, "p_value": p_value}


def romano_wolf_stepdown(
    returns_matrix: pd.DataFrame,
    benchmark: pd.Series,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    block: int = 21,
) -> pd.DataFrame:
    """Romano-Wolf stepdown correction for each strategy against a benchmark.

    A circular block bootstrap (default block length = 21 days) builds the
    joint null distribution. The observed and bootstrap statistics share
    the same studentized scale (t = mean / standard error). Stepdown
    procedure (Romano & Wolf 2005): at each step the maximum distribution
    is recomputed only from the strategies not yet rejected, then the
    largest hypothesis is tested; once one hypothesis is not rejected, the
    remaining hypotheses are also not rejected.
    """
    rng = np.random.default_rng(seed)
    merged = returns_matrix.join(benchmark.rename("benchmark"), how="inner").dropna()
    names = list(returns_matrix.columns)
    n = len(merged)
    diff = {
        c: merged[c].to_numpy(dtype=float) - merged["benchmark"].to_numpy(dtype=float)
        for c in names
    }

    def studentized(sample: np.ndarray) -> float:
        se = float(np.std(sample, ddof=1) / np.sqrt(len(sample)))
        return float(sample.mean() / se) if se > 0 else 0.0

    stat = {c: studentized(diff[c]) for c in names}
    # Studentized null distribution per strategy: (n_bootstrap, n_strategies).
    boot = np.empty((n_bootstrap, len(names)), dtype=float)
    for r in range(n_bootstrap):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
        for j, c in enumerate(names):
            boot[r, j] = studentized(diff[c][idx] - diff[c].mean())
    # Stepdown: critical values are recomputed from the remaining hypotheses.
    order = sorted(range(len(names)), key=lambda j: stat[names[j]], reverse=True)
    rows: list[dict[str, object]] = []
    remaining = list(order)
    step = 1
    while remaining:
        threshold = float(np.quantile(boot[:, remaining].max(axis=1), 1.0 - alpha))
        j = remaining[0]
        c = names[j]
        reject = bool(stat[c] > threshold)
        rows.append(
            {
                "strategy": c,
                "t_stat": stat[c],
                "critical_value": threshold,
                "reject": reject,
                "step": step,
            }
        )
        if not reject:
            break
        remaining = remaining[1:]
        step += 1
    return pd.DataFrame(rows)
