#!/usr/bin/env python3
"""Stage 5: risk, significance, and regime-robustness evaluation.

Input: stage-4 outputs (`portfolio_returns.csv`,
`baseline_returns.csv`, `weights.csv`).
Outputs (`--out`, default `reports/`):
- `metrics_table.csv`    : metrics per MV configuration (full and OOS)
- `regime_table.csv`     : metrics per regime (COVID 2020, 2021-2025)
- `seed_spread.csv`      : Sharpe spread across seeds per (k, estimator)
- `significance.csv`     : HAC and Romano-Wolf tests against the benchmarks
- `dsr_pbo.json`         : DSR of the best strategy and PBO of the main grid
- `figures/`             : equity curves, drawdown, Sharpe comparison

Decision log: docs/keputusan_desain.md sections E and H.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.evaluation.dsr_pbo import deflated_sharpe, pbo_cscv
from lq45.evaluation.metrics import (
    daily_risk_free_rate,
    rank_ic_and_spread,
    summarize_series,
)
from lq45.evaluation.regimes import split_regimes
from lq45.evaluation.significance import (
    mean_difference_test,
    romano_wolf_stepdown,
    sharpe_difference_test_lw,
)
from lq45.utils.config import RAW_DIR, REPORT_DIR, git_sha, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stage 5 portfolio evaluation.")
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="stage-3 predictions.csv; enables the Rank IC / spread table",
    )
    return parser.parse_args()


def config_key(frame: pd.DataFrame) -> pd.Series:
    """Text configuration key from the grid columns."""
    return (
        frame["k"].astype(str)
        + "|"
        + frame["estimator"]
        + "|L"
        + frame["lookback"].astype(str)
        + "|m"
        + frame["max_weight"].astype(str)
        + "|"
        + frame["seed_mode"]
    )


def mean_turnover(weights: pd.DataFrame, key: str) -> float:
    """Mean turnover per rebalancing for a single configuration.

    Turnover = sum of |weight differences| / 2 between consecutive
    rebalancing dates (one-way, as a fraction of the portfolio).
    """
    slice = weights[weights["config"] == key].copy()
    if slice.empty:
        return 0.0
    pivot = slice.pivot_table(
        index="date", columns="ticker", values="weight", fill_value=0.0
    ).sort_index()
    if len(pivot) < 2:
        return 0.0
    diff = pivot.diff().iloc[1:].abs().sum(axis=1) / 2.0
    return float(diff.mean())


def main() -> int:
    """Compute metrics, tests, and figures, then write the outputs."""
    args = parse_args()
    exp_cfg = load_config("experiment")
    periods_per_year = int(exp_cfg["evaluation"]["annualization"])
    regime_cfg = {k: v for k, v in exp_cfg["regimes"].items() if v.get("oos", False)}
    out_dir = args.out or REPORT_DIR
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    portfolio = pd.read_csv(args.portfolio, parse_dates=["date"])
    baseline = pd.read_csv(args.baselines, parse_dates=["date"])
    weights = pd.read_csv(args.weights, parse_dates=["date"])
    portfolio["config"] = config_key(portfolio)
    date = pd.DatetimeIndex(sorted(portfolio["date"].unique()))
    rf = daily_risk_free_rate(
        date, str(RAW_DIR / "macro" / "bi_7drrr.csv"), periods_per_year
    )
    rf.index = date
    rows: list[dict] = []
    series: dict[str, pd.Series] = {}
    weights["config"] = config_key(weights)
    for key, group in portfolio.groupby("config"):
        group = group.sort_values("date")
        returns = group.set_index("date")["return"]
        series[key] = returns
        summary = summarize_series(
            returns, rf.reindex(returns.index).fillna(0.0), periods_per_year
        )
        summary["turnover"] = mean_turnover(weights, key)
        summary["config"] = key
        summary.update(
            {
                "k": group["k"].iloc[0],
                "estimator": group["estimator"].iloc[0],
                "lookback": group["lookback"].iloc[0],
                "max_weight": group["max_weight"].iloc[0],
                "seed_mode": group["seed_mode"].iloc[0],
            }
        )
        rows.append(summary)
    metrics = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    metrics.to_csv(out_dir / "metrics_table.csv", index=False)
    # Rank quality of the stage-3 predictions, if they were provided.
    if args.predictions is not None:
        predictions = pd.read_csv(args.predictions, parse_dates=["date"])
        predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
        pred = predictions.groupby(["date", "ticker"], as_index=False)[
            ["pred_raw", "true_raw"]
        ].mean()
        ranker = rank_ic_and_spread(pred)
        pd.DataFrame([ranker]).to_csv(out_dir / "ranker_table.csv", index=False)
        print(
            f"ranker: Rank IC {ranker['rank_ic_mean']:.4f} "
            f"(IR {ranker['rank_ic_ir']:.3f}), top-minus-bottom "
            f"{ranker['top_minus_bottom_mean']:.5f} over "
            f"{int(ranker['n_dates'])} dates",
            flush=True,
        )
    else:
        print("ranker table skipped (no --predictions)", flush=True)
    # OOS regime per configuration.
    regime_rows: list[dict] = []
    for key, returns in series.items():
        frame = pd.DataFrame({"date": returns.index, "return": returns.to_numpy()})
        for name, slice in split_regimes(frame, "date", regime_cfg).items():
            if slice.empty:
                continue
            regime_idx = pd.DatetimeIndex(pd.to_datetime(slice["date"]))
            summary = summarize_series(
                slice.set_index("date")["return"],
                rf.reindex(regime_idx).fillna(0.0),
                periods_per_year,
            )
            summary["config"] = key
            summary["regime"] = name
            regime_rows.append(summary)
    pd.DataFrame(regime_rows).to_csv(out_dir / "regime_table.csv", index=False)
    # Cross-seed spread per (k, estimator) on the main grid.
    main = metrics[(metrics["lookback"] == 120) & (metrics["max_weight"] == 0.35)]
    spread = (
        main.groupby(["k", "estimator"])["sharpe"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    spread.to_csv(out_dir / "seed_spread.csv", index=False)
    # Significance: top configurations against 1/N and IHSG.
    baseline_idx = baseline.set_index("date")
    benchmark_1n = baseline_idx["ret_1n_full"]
    benchmark_ihsg = baseline_idx["ret_ihsg"]
    test_rows: list[dict] = []
    for key in metrics.head(args.top_n)["config"]:
        returns = series[key]
        for benchmark_name, benchmark in (
            ("1N", benchmark_1n),
            ("IHSG", benchmark_ihsg),
        ):
            result = sharpe_difference_test_lw(returns, benchmark, periods_per_year)
            result["test"] = "sharpe_lw2008"
            result["config"] = key
            result["benchmark"] = benchmark_name
            test_rows.append(result)
            supplementary = mean_difference_test(returns, benchmark, periods_per_year)
            supplementary["test"] = "mean_hac"
            supplementary["config"] = key
            supplementary["benchmark"] = benchmark_name
            test_rows.append(supplementary)
    rw_entries = main[main["seed_mode"] == "ensemble"]
    rw_matrix = pd.DataFrame({c: series[c] for c in rw_entries["config"]})
    rw_table = romano_wolf_stepdown(rw_matrix, benchmark_1n, seed=0)
    rw_table["benchmark"] = "1N"
    pd.concat(
        [pd.DataFrame(test_rows), rw_table], ignore_index=True, sort=False
    ).to_csv(out_dir / "significance.csv", index=False)
    # DSR of the best strategy and PBO over the full ensemble grid.
    best = metrics.iloc[0]["config"]
    n_trials = float(len(metrics))
    trial_srs = []
    for values in series.values():
        sd = float(values.std(ddof=1))
        trial_srs.append(float(values.mean() / sd) if sd > 0 else 0.0)
    dsr = deflated_sharpe(series[best], int(n_trials), trial_srs=trial_srs)
    pbo_matrix = pd.DataFrame(
        {c: series[c] for c in metrics[metrics["seed_mode"] == "ensemble"]["config"]}
    )
    pbo = pbo_cscv(pbo_matrix, n_groups=8)
    (out_dir / "dsr_pbo.json").write_text(
        json.dumps(
            {
                "best_config": best,
                "n_trials": n_trials,
                "dsr": dsr,
                "pbo": pbo,
                "git_sha": git_sha(),
                "created_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Figures: top-5 equity + benchmarks, and the main-grid Sharpe.
    equity = pd.DataFrame({c: (1.0 + series[c]).cumprod() for c in series})
    top5 = metrics.head(5)["config"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in top5:
        ax.plot(equity.index, equity[c], label=c, linewidth=1.0)
    ax.plot(
        baseline_idx.index,
        (1.0 + baseline_idx["ret_1n_full"]).cumprod(),
        label="1/N",
        linestyle="--",
    )
    ax.plot(
        baseline_idx.index,
        (1.0 + baseline_idx["ret_ihsg"]).cumprod(),
        label="IHSG",
        linestyle=":",
    )
    ax.set_title("Equity curves: top 5 configurations and benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital growth (x)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "equity_top5.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ens120 = main[main["seed_mode"] == "ensemble"].sort_values(["k", "estimator"])
    ax.bar(range(len(ens120)), ens120["sharpe"])
    ax.set_xticks(range(len(ens120)))
    ax.set_xticklabels(
        ens120["k"].astype(str) + "/" + ens120["estimator"], rotation=45, fontsize=7
    )
    ax.set_title("Ensemble main-grid Sharpe (L120, maxw 0.35)")
    ax.set_ylabel("Annualized Sharpe")
    fig.tight_layout()
    fig.savefig(figure_dir / "sharpe_grid_main.png", dpi=150)
    plt.close(fig)
    print(f"done: {len(metrics)} configurations -> {out_dir}", flush=True)
    print(f"best: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
