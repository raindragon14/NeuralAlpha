"""Rank stocks from ensemble predictions for top-k preselection.

Basis (details: docs/DECISIONS.md):
- Preselect then optimize: Wang et al. (2020) use deep learning for
  preselection before portfolio construction; Huang et al. (2024) use a
  two-stage approach with prescreening before Global Minimum Variance.
- The grid k = 5, 7, 10 answers RQ2; Chaweewanchon & Chaysiri (2022)
  test N = 5-10, Paiva et al. (2019) use 7, Wang et al. (2020) use 10.
- Averaging the ensemble across seeds dampens training variance; the
  spread is reported following Reimers & Gurevych (2017) and
  Bouthillier et al. (2021).
"""

from __future__ import annotations

import pandas as pd


def ensemble_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Average pred_raw across seeds per (date, ticker).

    The input uses columns `date, ticker, pred_raw` (per seed); the
    output has one row per (date, ticker) with a `pred_ens` column.
    """
    merged = (
        frame.groupby(["date", "ticker"], as_index=False)["pred_raw"]
        .mean()
        .rename(columns={"pred_raw": "pred_ens"})
    )
    return merged


def top_k(
    frame: pd.DataFrame, date: str, k: int, column: str = "pred_ens"
) -> list[str]:
    """Select the k highest-ranked stocks on a single date.

    Stocks without a prediction on that date (e.g. a WIKA suspension)
    are excluded; if there are fewer than k candidates, return those
    available.
    """
    subset = frame[frame["date"] == date].dropna(subset=[column])
    subset = subset.sort_values([column, "ticker"], ascending=[False, True])
    return subset["ticker"].head(k).tolist()
