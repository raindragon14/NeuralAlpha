"""Portfolio evaluation metrics: risk, return, and implicit costs.

Basis (details: docs/keputusan_desain.md):
- Sharpe, Sortino, Calmar, MDD, turnover: Malhotra et al. (2023) use
  Sharpe/Sortino/Omega as mutual-fund standards; Wang & Liu (2025)
  define risk-sensitive evaluation (Sharpe/Sortino/Calmar).
- Daily risk-free = annual BI-7DRRR divided by 252 (E8, official BI source).
- Annualization over 252 trading days (E9, standard convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown_and_duration(equity: pd.Series) -> tuple[float, int]:
    """Maximum drawdown (fraction) and day count from peak to trough."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0.0, np.nan)
    drawdown = drawdown.fillna(0.0)
    mdd = float(drawdown.min())
    trough_position = int(drawdown.to_numpy().argmin()) if len(drawdown) else 0
    peak_position = (
        int(peak.to_numpy()[: trough_position + 1].argmax()) if len(drawdown) else 0
    )
    return mdd, max(0, trough_position - peak_position)


def summarize_series(
    returns: pd.Series,
    risk_free: pd.Series | float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Summarize a daily return series into annualized metrics.

    `risk_free` is a daily series or a fixed number. Sortino uses a
    target of zero on the downside. Calmar uses annual return divided
    by |MDD| (zero when MDD is zero).
    """
    r = returns.to_numpy(dtype=float)
    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(returns.index).fillna(0.0).to_numpy(dtype=float)
    else:
        rf = np.full_like(r, float(risk_free))
    excess = r - rf
    n = len(r)
    cumulative = float(np.prod(1.0 + r) - 1.0) if n else 0.0
    annual = float((1.0 + cumulative) ** (periods_per_year / n) - 1.0) if n else 0.0
    vol = float(np.std(excess, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / vol) if vol > 0 else 0.0
    downside = np.minimum(excess, 0.0)
    vol_down = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        float(np.sqrt(periods_per_year) * excess.mean() / vol_down)
        if vol_down > 0
        else 0.0
    )
    equity = pd.Series(np.cumprod(1.0 + r), index=returns.index)
    mdd, _ = max_drawdown_and_duration(equity)
    calmar = float(annual / abs(mdd)) if mdd < 0 else 0.0
    return {
        "cumulative_return": cumulative,
        "annual_return": annual,
        "annual_vol": float(vol * np.sqrt(periods_per_year)) if n > 1 else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "n_days": float(n),
    }


def rank_ic_and_spread(
    predictions: pd.DataFrame, top_frac: float = 1.0 / 3.0
) -> dict[str, float]:
    """Cross-sectional rank quality of the predictions.

    Stage 4 only uses the ordering of the predictions, so the model is
    scored as a ranker: per date, the Spearman correlation between
    `pred_raw` and `true_raw` (the Rank IC), and the realized return of the
    top `top_frac` minus the bottom `top_frac` ranked by prediction.

    `predictions` needs columns `date, ticker, pred_raw, true_raw`. Dates
    with fewer than 3 stocks, or with a degenerate (constant) prediction
    column, are skipped.
    """
    rank_ics: list[float] = []
    spreads: list[float] = []
    for _, group in predictions.groupby("date"):
        group = group.dropna(subset=["pred_raw", "true_raw"])
        n = len(group)
        if n < 3:
            continue
        ic = float(group["pred_raw"].corr(group["true_raw"], method="spearman"))
        if not np.isfinite(ic):
            continue
        q = max(1, int(round(n * top_frac)))
        ordered = group.sort_values("pred_raw")
        rank_ics.append(ic)
        spreads.append(
            float(
                ordered["true_raw"].tail(q).mean() - ordered["true_raw"].head(q).mean()
            )
        )
    if not rank_ics:
        raise ValueError("no date with at least 3 complete predictions")
    ic = np.asarray(rank_ics, dtype=float)
    spread = np.asarray(spreads, dtype=float)
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    return {
        "rank_ic_mean": float(ic.mean()),
        "rank_ic_std": ic_std,
        "rank_ic_ir": float(ic.mean() / ic_std) if ic_std > 0 else 0.0,
        "top_minus_bottom_mean": float(spread.mean()),
        "top_minus_bottom_std": float(spread.std(ddof=1)) if len(spread) > 1 else 0.0,
        "n_dates": float(len(ic)),
    }


def daily_risk_free_rate(
    date: pd.DatetimeIndex, rate_file: str, periods_per_year: int = 252
) -> pd.Series:
    """Daily risk-free series from the BI-7DRRR decision history.

    The file uses columns `date, rate` (annual percent on the decision
    date); values are forward-filled and then divided by 100 and 252.
    """
    rate = pd.read_csv(rate_file, parse_dates=["date"]).set_index("date")
    rate.index = pd.to_datetime(rate.index)
    daily = rate["rate"].reindex(date, method="ffill").bfill()
    return (daily / 100.0 / periods_per_year).astype(float)
