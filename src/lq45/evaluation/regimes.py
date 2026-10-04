"""Market-regime splits for strategy robustness tests.

Basis (details: docs/DECISIONS.md):
- COVID regime test: Huang et al. (2024) test robustness during COVID.
- OOS regimes: COVID 2020 and the 2021-2025 recovery/rate-hike period;
  the 2018-2019 design period is calm and is not used for evaluation
  (E7, configs/experiment.yaml `regimes`).
"""

from __future__ import annotations

import pandas as pd


def split_regimes(
    frame: pd.DataFrame,
    date_column: str,
    regimes: dict[str, dict[str, str]],
) -> dict[str, pd.DataFrame]:
    """Split a date-columned frame into each regime.

    Bounds use inclusive `YYYY-MM-DD` strings from the configuration.
    """
    date = pd.to_datetime(frame[date_column])
    result: dict[str, pd.DataFrame] = {}
    for name, bounds in regimes.items():
        start = pd.Timestamp(bounds["start"])
        end = pd.Timestamp(bounds["end"])
        mask = (date >= start) & (date <= end)
        result[name] = frame[mask].reset_index(drop=True)
    return result
