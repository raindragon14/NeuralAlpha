"""Deflated Sharpe Ratio and Probability of Backtest Overfitting.

Basis (details: docs/DECISIONS.md):
- DSR: Bailey & Lopez de Prado (2014) correct the Sharpe ratio for the
  number of trials and non-normality (skew, kurtosis).
- PBO/CSCV: Bailey et al. (2016) measure the probability that the
  selected strategy is overfit via combinatorial symmetric
  cross-validation.
- Both answer E16 (backtest overfitting control) together with
  Romano & Wolf (2005).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe with a zero risk-free rate (internal comparison)."""
    r = returns.to_numpy(dtype=float)
    vol = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    if vol <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / vol)


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """Expected maximum Sharpe ratio after n_trials independent trials.

    Bailey & Lopez de Prado (2014) Eq. (1)-(2), reference implementation
    `getExpMaxSR(mu, sigma, numTrials) = mu + sigma * maxZ`: `sigma =
    sqrt(variance)` is the standard deviation of the TRIALS' Sharpe
    ratios (not the standard error of one estimate) and scales the whole
    maxZ bracket.
    """
    if n_trials < 2 or variance <= 0:
        return 0.0
    gamma = 0.5772156649
    max_z = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials) + gamma * norm.ppf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return float(math.sqrt(variance) * max_z)


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    trial_srs: list[float] | None = None,
    benchmark: float = 0.0,
) -> dict[str, float]:
    """DSR: probability the true Sharpe exceeds SR0 after trial correction.

    Bailey & Lopez de Prado (2014): DSR = Phi((SR - SR0) / sqrt(V[SR]))
    with per-period (non-annualized) Sharpe ratios throughout, V[SR]
    carrying the skewness/kurtosis correction, and SR0 scaled by the
    variance of the trials' Sharpe ratios (`trial_srs`); V[SR] is used as
    a conservative fallback when the trials are unknown. `benchmark` is
    an extra per-period hurdle (e.g. a benchmark's Sharpe ratio).
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n < 3 or n_trials < 1:
        return {"sharpe": 0.0, "benchmark": float(benchmark), "dsr": 0.5}
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return {"sharpe": 0.0, "benchmark": float(benchmark), "dsr": 0.5}
    sr = float(np.mean(r) / sd)
    skew = float(pd.Series(r).skew())
    kurtosis_pearson = float(pd.Series(r).kurtosis()) + 3.0
    var_sr = max(
        (1.0 - skew * sr + (kurtosis_pearson - 1.0) / 4.0 * sr * sr) / (n - 1),
        1e-12,
    )
    if trial_srs is not None and len(trial_srs) >= 2:
        var_trials = float(np.var(np.asarray(trial_srs, dtype=float), ddof=1))
    else:
        var_trials = var_sr
    sr0 = max(float(benchmark), expected_max_sharpe(int(n_trials), var_trials))
    return {
        "sharpe": sr,
        "benchmark": float(sr0),
        "dsr": float(norm.cdf((sr - sr0) / math.sqrt(var_sr))),
    }


def pbo_cscv(returns_matrix: pd.DataFrame, n_groups: int = 8) -> dict[str, float]:
    """PBO via CSCV: probability the in-sample choice loses out-of-sample.

    The matrix has strategies as columns and dates as rows. Rows are
    split into `n_groups` equal-length segments; each fold uses half the
    segments as in-sample. The best in-sample strategy is compared with
    its out-of-sample rank; PBO = probability of a negative logit
    (Bailey et al. 2016).
    """
    frame = returns_matrix.dropna()
    n = len(frame)
    n_strat = len(frame.columns)
    segments = np.array_split(np.arange(n), n_groups)
    half = n_groups // 2
    logit: list[float] = []
    for in_combination in itertools.combinations(range(n_groups), half):
        in_groups = list(in_combination)
        out_groups = [s for s in range(n_groups) if s not in in_groups]
        in_idx = np.concatenate([segments[s] for s in in_groups])
        out_idx = np.concatenate([segments[s] for s in out_groups])
        sr_in = frame.iloc[in_idx].apply(annualized_sharpe)
        sr_out = frame.iloc[out_idx].apply(annualized_sharpe)
        out_ranks = sr_out.rank(ascending=False)
        best = sr_in.idxmax()
        position = float(out_ranks[best])
        # Positive logit when the in-sample best ranks above the
        # out-of-sample median; PBO = probability of a negative logit.
        logit.append(math.log((n_strat + 1.0 - position) / position))
    logit_arr = np.array(logit, dtype=float)
    return {
        "pbo": float((logit_arr < 0).mean()) if len(logit_arr) else 0.5,
        "n_splits": float(len(logit_arr)),
        "mean_logit": float(logit_arr.mean()) if len(logit_arr) else 0.0,
    }
