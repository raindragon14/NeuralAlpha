#!/usr/bin/env python3
"""Stage 4: Top-K ranking and Mean-Variance optimization.

Input: `predictions.csv` from stage 3 (all rows are out-of-sample; the
`role` column is kept for audit only).
Outputs per run (`experiments/<run_id>/`):
    - `weights.csv`            : rebalancing weights per date and configuration
    - `portfolio_returns.csv`  : equity and daily net return per configuration
    - `baseline_returns.csv`   : 1/N pool and IHSG buy-and-hold
    - `run_info.json`          : execution record for auditing

Grid-search configuration (see `configs/portfolio.yaml`):
    - k: values {5, 7, 10, 3, 15, 20}, L: {120, 60, 252},
      max_weight: {0.35, 0.25, 0.50, 1.0}, and the ensemble as well as
      per-seed ranking modes. The `--grid main` option gives a fast
      run (k {5, 7, 10}, L 120, max_weight 0.35).

Methodology:
    Top-k selection uses the CNN-BiLSTM predictions. Mean-Variance
    optimization minimizes variance subject to a target return equal
    to the mean pred_ens (Chaweewanchon & Chaysiri 2022, Section 3.1).
    GMV is not used.

Decision log: docs/DECISIONS.md sections E and H.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.portfolio.backtest import run_backtest
from lq45.portfolio.costs import target_shares, transaction_value
from lq45.portfolio.ranking import ensemble_predictions
from lq45.utils.config import (
    EXPERIMENT_DIR,
    INTERIM_DIR,
    RAW_DIR,
    ensure_dirs,
    git_sha,
    load_config,
)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the optimization configuration.

    Returns:
        Namespace with the parameters: predictions path, output directory,
        grid scope, initial capital, and seed ranking mode.
    """
    parser = argparse.ArgumentParser(
        description="Top-k ranking and portfolio optimization."
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="predictions.csv from stage 3"
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--grid",
        choices=["main", "full"],
        default="full",
        help="grid scope (default: full)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000_000.0,
        help="initial capital in rupiah (default: 100 million)",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["ensemble", "all"],
        default="all",
        help="rank the ensemble only, or also per seed",
    )
    return parser.parse_args()


def load_prices() -> pd.DataFrame:
    """Load the adjusted closing prices (Adj Close) of every LQ45 stock.

    Each CSV file in the interim directory represents one ticker.
    The index is dates and the columns are tickers with the `.JK` suffix.

    Returns:
        DataFrame with daily prices sorted by date.
    """
    frames: dict[str, pd.Series] = {}
    for csv_path in sorted((INTERIM_DIR / "prices_clean").glob("*.csv")):
        ticker = csv_path.stem + ".JK"
        frame = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
        if "Adj Close" not in frame.columns:
            raise ValueError(f"{csv_path} is missing the 'Adj Close' column")
        frames[ticker] = frame["Adj Close"]
    prices = pd.DataFrame(frames).sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices


def prediction_frame(path: Path) -> pd.DataFrame:
    """Load the stage-3 prediction results.

    Every surviving `(date, ticker, seed)` row is an out-of-sample
    prediction (the stage-3 dedup keeps exactly one row per key, made by a
    model trained strictly before that date), so no rows are dropped here
    and the `role` column is kept for audit only. Keeping the
    `role == "validation"` rows of fold 0 restores the first out-of-sample
    window (Jan-Mar 2020) that would otherwise be missing from the
    evaluation, including the COVID crash.

    Args:
        path: Path to the `predictions.csv` file.

    Returns:
        DataFrame with the out-of-sample predictions and its index reset.
    """
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = {"date", "ticker", "pred_raw"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.reset_index(drop=True)


def ranking_frame(frame: pd.DataFrame, seed_mode: str) -> dict[str, pd.DataFrame]:
    """Build the prediction frames per ensemble mode and per seed.

    Each mode maps to a DataFrame with the columns `date`, `ticker`, and
    `pred_ens`. The `ensemble` mode averages all seeds; the individual-seed
    mode produces one DataFrame per seed for reporting the prediction spread.

    Args:
        frame: DataFrame produced by `prediction_frame`.
        seed_mode: `"ensemble"` averages all seeds; any other value keeps
            one frame per seed.

    Returns:
        Dictionary mapping the mode name to a ranking DataFrame.
    """
    ranking_frames: dict[str, pd.DataFrame] = {}
    ranking_frames["ensemble"] = ensemble_predictions(frame)
    if seed_mode == "all":
        for seed in sorted(frame["seed"].unique().tolist()):
            subset = frame[frame["seed"] == seed][["date", "ticker", "pred_raw"]]
            ranking_frames[f"seed{seed}"] = subset.rename(
                columns={"pred_raw": "pred_ens"}
            )
    return ranking_frames


def baselines(
    prices: pd.DataFrame,
    oos_dates: list[str],
    capital: float,
    buy_fee: float = 0.0019,
    sell_fee: float = 0.0029,
    lot: int = 100,
    step: int = 21,
) -> pd.DataFrame:
    """Compute the daily returns of the fee-paying 1/N and IHSG benchmarks.

    1/N rebalances to equal weights on the same `step`-day schedule as the
    strategies (same 1-day execution lag, lot rounding and buy/sell fees),
    so the comparison is like-for-like; DeMiguel et al. (2009) use a
    periodically rebalanced 1/N. IHSG is buy-and-hold normalized to the
    initial capital on the first OOS date (price index without dividends).

    Args:
        prices: Daily closing-price DataFrame.
        oos_dates: List of out-of-sample dates.
        capital: Initial capital in rupiah.
        buy_fee: Buy fee as a fraction of the buy value.
        sell_fee: Sell fee as a fraction of the sell value.
        lot: Lot size for share rounding.
        step: Trading days between rebalancings.

    Returns:
        DataFrame with the equity and daily returns of both benchmarks.
    """
    dates = pd.to_datetime(sorted(set(oos_dates)))
    prices_oos = prices.reindex(dates).ffill()
    cash = float(capital)
    holdings = pd.Series(dtype=float)
    equities: list = [None] * len(dates)

    def mark(pos: int) -> None:
        day_prices = prices_oos.iloc[pos]
        holdings_value = float(
            (holdings * day_prices.reindex(holdings.index).fillna(0.0)).sum()
            if len(holdings)
            else 0.0
        )
        equities[pos] = cash + holdings_value

    for start in range(0, len(dates), step):
        end = min(start + step, len(dates))
        execution = min(start + 1, len(dates) - 1)
        for pos in range(start, min(execution, end)):
            mark(pos)
        exec_prices = prices_oos.iloc[execution].dropna()
        available = exec_prices[exec_prices > 0].index
        if len(available):
            equity = cash + float(
                (holdings * exec_prices.reindex(holdings.index).fillna(0.0)).sum()
                if len(holdings)
                else 0.0
            )
            weights = pd.Series(1.0 / len(available), index=available)
            target = target_shares(weights, exec_prices, equity, lot)
            target = target.reindex(holdings.index.union(target.index)).fillna(0.0)
            old = holdings.reindex(target.index).fillna(0.0)
            cash_flow, fee = transaction_value(
                old, target, exec_prices, buy_fee, sell_fee
            )
            cash = cash + cash_flow - fee
            holdings = target
        for pos in range(execution, end):
            mark(pos)
    equity_1n = pd.Series(equities, index=dates, dtype=float).ffill()

    ihsg = pd.read_csv(RAW_DIR / "benchmark_ihsg.csv", parse_dates=["Date"]).set_index(
        "Date"
    )
    ihsg.index = pd.to_datetime(ihsg.index)
    ihsg_oos = ihsg[ihsg.index.isin(dates)].sort_index()["Adj Close"]
    equity_ihsg = ihsg_oos / ihsg_oos.iloc[0] * capital
    merged = pd.DataFrame({"eq_1n_full": equity_1n, "eq_ihsg": equity_ihsg})
    merged.index.name = "date"
    merged = merged.reset_index()
    merged["ret_1n_full"] = merged["eq_1n_full"].pct_change().fillna(0.0)
    merged["ret_ihsg"] = merged["eq_ihsg"].pct_change().fillna(0.0)
    return merged


def main() -> int:
    """Run the portfolio-optimization grid search and write its results.

    Workflow:
        1. Load the configuration, prices, and predictions.
        2. Build the grid combinations (k, estimator, lookback, max_weight).
        3. Run the backtest per combination, appending results to CSV as
           it goes for resilience against failures.
        4. Compute the 1/N and IHSG baselines.
        5. Write the execution metrics to `run_info.json`.

    Returns:
        Exit code 0 on success.
    """
    args = parse_args()
    port_cfg = load_config("portfolio")
    data_cfg = load_config("data")
    opt_cfg = port_cfg["optimization"]
    start = time.time()
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out or (EXPERIMENT_DIR / f"optimize_{stamp}")
    ensure_dirs(run_dir)
    if args.grid == "main":
        k_list = list(port_cfg["selection"]["k_values"])
        l_list = [int(port_cfg["covariance"]["lookback_days"])]
        maxw_list = [float(port_cfg["constraints"]["max_weight"])]
    else:
        k_list = list(port_cfg["selection"]["k_values"]) + list(
            port_cfg["selection"]["k_sensitivity"]
        )
        l_list = [int(port_cfg["covariance"]["lookback_days"])] + list(
            port_cfg["covariance"]["lookback_sensitivity"]
        )
        maxw_list = [float(port_cfg["constraints"]["max_weight"])] + [
            float(x) for x in port_cfg["constraints"]["max_weight_sensitivity"]
        ]
    estimators = list(port_cfg["covariance"]["estimators"])
    optimizer_types = opt_cfg.get("types", ["target_return"])
    costs = data_cfg["costs"]
    frame = prediction_frame(args.predictions)
    prices = load_prices()
    ranking = ranking_frame(frame, args.seed_mode)
    oos_dates = sorted(frame["date"].unique().tolist())
    total = (
        len(k_list)
        * len(estimators)
        * len(l_list)
        * len(maxw_list)
        * len(ranking)
        * len(optimizer_types)
    )
    progress = 0
    for optimizer_type in optimizer_types:
        for mode_name, ranking_df in ranking.items():
            for k in k_list:
                for est in estimators:
                    for length in l_list:
                        for maxw in maxw_list:
                            progress += 1
                            weights, value = run_backtest(
                                ranking_df,
                                prices,
                                int(k),
                                est,
                                int(length),
                                float(maxw),
                                capital=args.capital,
                                buy_fee=float(costs["buy_fee"]),
                                sell_fee=float(costs["sell_fee"]),
                                lot=int(costs["lot_size"]),
                                step=int(port_cfg.get("rebalance_days", 21)),
                                ridge_epsilon=float(
                                    port_cfg["covariance"]["ridge_epsilon"]
                                ),
                                optimizer_type=optimizer_type,
                                execution_lag_days=int(
                                    port_cfg.get("execution_lag_days", 1)
                                ),
                            )
                            weights["seed_mode"] = mode_name
                            value["k"] = int(k)
                            value["estimator"] = est
                            value["lookback"] = int(length)
                            value["max_weight"] = float(maxw)
                            value["seed_mode"] = mode_name
                            # Append the results of each iteration to
                            # prevent data loss if the process fails
                            # partway through.
                            weights_path = run_dir / "weights.csv"
                            returns_path = run_dir / "portfolio_returns.csv"
                            weights.to_csv(
                                weights_path,
                                mode="a",
                                header=not weights_path.exists(),
                                index=False,
                            )
                            value.to_csv(
                                returns_path,
                                mode="a",
                                header=not returns_path.exists(),
                                index=False,
                            )
                            if progress % 100 == 0 or progress == total:
                                print(
                                    f"[{progress}/{total}] {optimizer_type} "
                                    f"{mode_name} k={k} {est} L={length} maxw={maxw}",
                                    flush=True,
                                )
    baselines(
        prices,
        oos_dates,
        args.capital,
        buy_fee=float(costs["buy_fee"]),
        sell_fee=float(costs["sell_fee"]),
        lot=int(costs["lot_size"]),
        step=int(port_cfg.get("rebalance_days", 21)),
    ).to_csv(run_dir / "baseline_returns.csv", index=False)
    info = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "predictions": str(args.predictions),
        "grid": args.grid,
        "k_values": k_list,
        "estimators": estimators,
        "optimizer_types": optimizer_types,
        "lookbacks": l_list,
        "max_weights": maxw_list,
        "seed_modes": sorted(ranking),
        "capital": args.capital,
        "n_configs": total,
        "wall_sec": round(time.time() - start, 1),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"done: {total} configurations -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
