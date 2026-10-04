#!/usr/bin/env python3
"""Stage 2: clean the raw data, then build features and targets.

Outputs:
- ``data/interim/prices_clean/<TICKER>.csv``: OHLCV aligned to the exchange calendar
- ``data/processed/features/<TICKER>.csv``  : 8 features + the ``target`` column

Feature specification: configs/experiment.yaml
Decision log: docs/DECISIONS.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from lq45.features import build_macro, build_panel, clean_prices
from lq45.utils.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
    load_config,
    stock_file_name,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build features and targets from the raw data."
    )
    return parser.parse_args()


def main() -> int:
    """Build features and targets for every candidate stock."""
    parse_args()

    data_cfg = load_config("data")
    exp_cfg = load_config("experiment")
    model_cfg = load_config("model")
    tickers = list(load_config("universe")["tickers"])

    ffill_days = int(data_cfg["cleaning"]["forward_fill_max_days"])
    horizon = int(model_cfg["horizon_days"])
    periods = exp_cfg["features"]["indicators"]
    macro_transform = exp_cfg["features"]["macro_transform"]

    # The exchange calendar is taken from IHSG.
    ihsg = pd.read_csv(RAW_DIR / "benchmark_ihsg.csv", parse_dates=["Date"])
    calendar = pd.DatetimeIndex(ihsg["Date"]).sort_values()
    print(
        f"Exchange calendar : {len(calendar)} days"
        f" ({calendar.min().date()} to {calendar.max().date()})"
    )

    jisdor = pd.read_csv(RAW_DIR / "macro" / "jisdor.csv", parse_dates=["date"])
    bi_rate = pd.read_csv(RAW_DIR / "macro" / "bi_7drrr.csv", parse_dates=["date"])
    macro = build_macro(jisdor, bi_rate, calendar, macro_transform)

    raw: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = RAW_DIR / "prices" / f"{stock_file_name(ticker)}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        raw[ticker] = clean_prices(frame, calendar, ffill_days)

    if not raw:
        print(
            f"no price files found for the configured universe in {RAW_DIR / 'prices'};"
            " run 01_fetch_data.py first",
            file=sys.stderr,
        )
        return 1

    panel = build_panel(raw, calendar, macro, horizon, periods)

    interim_dir = INTERIM_DIR / "prices_clean"
    processed_dir = PROCESSED_DIR / "features"
    ensure_dirs(interim_dir, processed_dir)

    for ticker, clean in raw.items():
        clean.to_csv(interim_dir / f"{stock_file_name(ticker)}.csv")
    for ticker, features in panel.items():
        features.to_csv(processed_dir / f"{stock_file_name(ticker)}.csv")

    column = list(next(iter(panel.values())).columns)
    summary = pd.DataFrame(
        {
            "filled": {
                k: int(sum(features[k].notna().sum() for features in panel.values()))
                for k in column
            },
            "missing": {
                k: int(sum(features[k].isna().sum() for features in panel.values()))
                for k in column
            },
        }
    )
    summary["total"] = summary["filled"] + summary["missing"]
    complete_rows = int(
        sum(features[column].notna().all(axis=1).sum() for features in panel.values())
    )
    print(f"\nStocks processed : {len(panel)}")
    print(
        f"Complete rows    : {complete_rows} of {summary['total'].max()}"
        " (all features + target filled)"
    )
    print(f"Output columns   : {', '.join(column)}")
    print("\nPer-column summary (aggregated across stocks):")
    print(summary.to_string())
    print(f"\nInterim   : {interim_dir}")
    print(f"Processed : {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
