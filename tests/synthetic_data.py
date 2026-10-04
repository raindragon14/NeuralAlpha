"""Deterministic synthetic raw data for the offline pipeline smoke test.

Generates the same ``data/raw/`` layout that ``01_fetch_data.py`` would write,
so stages 02-05 can run without network access. Not real market data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TICKERS: tuple[str, ...] = (
    "AAAA.JK",
    "BBBB.JK",
    "CCCC.JK",
    "DDDD.JK",
    "EEEE.JK",
    "FFFF.JK",
)


def _ohlcv(
    rng: np.random.Generator, dates: pd.DatetimeIndex, start: float
) -> pd.DataFrame:
    """One synthetic OHLCV frame indexed by ``Date``."""
    steps = rng.normal(0.0004, 0.015, size=len(dates))
    close = start * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0.004, 0.0, size=len(dates)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0.0, 0.003, size=len(dates))),
            "High": close * (1 + spread),
            "Low": close * (1 - spread),
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, size=len(dates)).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def write_raw(root: Path, n_days: int = 760, seed: int = 0) -> Path:
    """Write a full ``raw/`` tree under ``root`` and return ``root``.

    ``n_days`` defaults to 760 business days (~3 years), enough to clear the
    2-year initial training window plus a validation and a test period.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")

    raw = root / "raw"
    prices_dir = raw / "prices"
    macro_dir = raw / "macro"
    prices_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)

    _ohlcv(rng, dates, start=6_000.0).to_csv(raw / "benchmark_ihsg.csv")
    for ticker in TICKERS:
        start = float(rng.integers(500, 5_000))
        _ohlcv(rng, dates, start=start).to_csv(
            prices_dir / f"{ticker.replace('.JK', '')}.csv"
        )

    pd.DataFrame({"date": dates[::21], "rate": 6.0}).to_csv(
        macro_dir / "jisdor.csv", index=False
    )
    pd.DataFrame({"date": dates[::63], "rate": 5.0}).to_csv(
        macro_dir / "bi_7drrr.csv", index=False
    )
    return root
