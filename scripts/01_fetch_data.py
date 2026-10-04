#!/usr/bin/env python3
"""Stage 1: download the raw data into ``data/raw/``.

Outputs:
- ``data/raw/prices/<TICKER>.csv``  : daily OHLCV (``Adj Close`` column)
- ``data/raw/macro/jisdor.csv``     : official USD/IDR reference rate (BI)
- ``data/raw/macro/bi_7drrr.csv``   : history of the BI-7DRRR decisions
- ``data/raw/benchmark_ihsg.csv``   : IHSG for the buy-and-hold benchmark
- ``data/raw/metadata.json``        : download summary (time, row counts)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from lq45.data import fetch_bi_rate, fetch_equities, fetch_jisdor
from lq45.utils.config import RAW_DIR, ensure_dirs, load_config, stock_file_name


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download raw LQ45 and macro data.")
    parser.add_argument("--start", default=None, help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="inclusive end date (YYYY-MM-DD)")
    parser.add_argument(
        "--tickers",
        default=None,
        help="comma-separated ticker list; defaults to configs/universe.yaml",
    )
    return parser.parse_args()


def main() -> int:
    """Download prices, exchange rates, policy rate, and benchmark to ``data/raw/``."""
    args = parse_args()
    data_cfg = load_config("data")

    start = args.start or data_cfg["period"]["start"]
    end = args.end or data_cfg["period"]["end"]
    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else list(load_config("universe")["tickers"])
    )

    # Yahoo uses an exclusive end bound.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    prices_dir = RAW_DIR / "prices"
    macro_dir = RAW_DIR / "macro"
    ensure_dirs(prices_dir, macro_dir)

    print(f"Period    : {start} to {end}")
    print(f"Tickers   : {len(tickers)} stocks")

    frames = fetch_equities(tickers, start, end_exclusive)
    saved, missing = [], []
    for ticker, frame in frames.items():
        if frame is None or frame.empty:
            missing.append(ticker)
            continue
        name = stock_file_name(ticker)
        frame.to_csv(prices_dir / f"{name}.csv")
        saved.append((name, len(frame)))

    print("\nSaved stock prices:")
    for name, n_rows in saved:
        print(f"  {name:<6} {n_rows:>5} rows")
    if missing:
        print(f"  MISSING DATA: {', '.join(missing)}")

    fx = fetch_jisdor(start, end)
    if not fx.empty:
        fx.to_csv(macro_dir / "jisdor.csv", index=False)
        print(f"JISDOR USD/IDR: {len(fx)} rows -> {macro_dir / 'jisdor.csv'}")
        print(
            f"  range {fx['date'].min().date()} to {fx['date'].max().date()}"
            f" | rate {fx['rate'].min():,.0f}-{fx['rate'].max():,.0f}"
        )

    bi = fetch_bi_rate()
    bi = bi[(bi["date"] >= pd.Timestamp(start)) & (bi["date"] <= pd.Timestamp(end))]
    bi.to_csv(macro_dir / "bi_7drrr.csv", index=False)
    print(f"BI-7DRRR     : {len(bi)} meetings -> {macro_dir / 'bi_7drrr.csv'}")
    if not bi.empty:
        print(
            f"  range {bi['date'].min().date()} to {bi['date'].max().date()}"
            f" | rate {bi['rate'].min():.2f}-{bi['rate'].max():.2f}%"
        )

    benchmark = data_cfg["sources"].get("benchmark", "^JKSE")
    benchmark_frame = fetch_equities([benchmark], start, end_exclusive)[benchmark]
    benchmark_rows = 0
    if benchmark_frame is not None and not benchmark_frame.empty:
        benchmark_frame.to_csv(RAW_DIR / "benchmark_ihsg.csv")
        benchmark_rows = len(benchmark_frame)
        print(
            f"\nBenchmark {benchmark:<6}: {benchmark_rows} rows"
            f" -> {RAW_DIR / 'benchmark_ihsg.csv'}"
        )

    metadata = {
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "period": {"start": start, "end": end},
        "tickers": list(tickers),
        "missing": missing,
        "rows": {name: n_rows for name, n_rows in saved},
        "benchmark": {"ticker": benchmark, "rows": benchmark_rows},
        "sources": {
            "prices": "Yahoo Finance (.JK, adjusted close)",
            "fx": "Bank Indonesia (JISDOR, official reference exchange rate)",
            "bi_rate": "Bank Indonesia (bi-rate.aspx)",
            "benchmark": "Yahoo Finance (^JKSE)",
        },
    }
    (RAW_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetadata     : {RAW_DIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
