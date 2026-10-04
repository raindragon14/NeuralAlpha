"""Indonesian retail transaction-cost simulation with 100-share lots.

Basis (details: docs/DECISIONS.md):
- Buy fee 0.19%, sell fee 0.29%, 100-share lot following Indonesian
  retail securities rules (A8).
- Monthly rebalancing every 21 trading days: Espiga-Fernandez et al.
  (2024) use cost-efficient periodic rebalancing; Huang et al. (2024)
  use a monthly horizon (E4).
"""

from __future__ import annotations

import math

import pandas as pd


def target_shares(
    weights: pd.Series,
    prices: pd.Series,
    equity: float,
    lot: int = 100,
) -> pd.Series:
    """Target number of shares per stock, floored to a multiple of the lot.

    Funds insufficient for one lot remain as cash.
    """
    shares: dict[str, int] = {}
    for ticker, w in weights.items():
        p = float(prices.get(ticker, float("nan")))
        if not math.isfinite(p) or p <= 0 or w <= 0:
            shares[ticker] = 0
            continue
        unit = int((float(w) * equity) // (p * lot))
        shares[ticker] = unit * lot
    return pd.Series(shares, dtype=float)


def transaction_value(
    old: pd.Series,
    new: pd.Series,
    prices: pd.Series,
    buy_fee: float = 0.0019,
    sell_fee: float = 0.0029,
) -> tuple[float, float]:
    """Trade value and total fee from the change in positions.

    Returns (net_cash_flow, fee). A positive net cash flow means sales
    exceed purchases before fees.
    """
    all_tickers = set(old.index) | set(new.index) | set(prices.index)
    buys = 0.0
    sells = 0.0
    for ticker in all_tickers:
        p = float(prices.get(ticker, 0.0) or 0.0)
        delta = float(new.get(ticker, 0.0)) - float(old.get(ticker, 0.0))
        if delta > 0:
            buys += delta * p
        elif delta < 0:
            sells += -delta * p
    fee = buys * buy_fee + sells * sell_fee
    return sells - buys, fee
