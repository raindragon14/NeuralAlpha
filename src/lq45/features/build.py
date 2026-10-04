"""Feature, indicator, macro, and target construction.

Source of each choice (details: docs/DECISIONS.md):
- `Adj Close` price level and volume: Sebastian & Tantia (2024); Sen & Dutta (2021).
- RSI, CCI, CMO, MFI: Espiga-Fernandez et al. (2024) Appendix B.
- BI-7DRRR level and JISDOR log-return: extension to fill the gap in
  Chaweewanchon & Chaysiri (2022).
- OHLC adjustment with the `Adj Close / Close` factor: principle to prevent
  jumps caused by dividends and stock splits.
- Exchange calendar from IHSG; forward-fill at most 3 days.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from lq45.features.indicators import cci, cmo, mfi, rsi

# Espiga-Fernandez et al. (2024) do not fix a period per indicator, so standard
# values are used: RSI 14, CCI 20, CMO 14, MFI 14.
INDICATOR_PERIODS: dict[str, int] = {"rsi": 14, "cci": 20, "cmo": 14, "mfi": 14}

PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close")


def clean_prices(
    frame: pd.DataFrame, calendar: pd.DatetimeIndex, ffill_days: int
) -> pd.DataFrame:
    """Align to the exchange calendar, then fill gaps for at most `ffill_days`."""
    frame = frame.reindex(calendar)
    frame[list(PRICE_COLUMNS)] = frame[list(PRICE_COLUMNS)].ffill(limit=ffill_days)
    frame["Volume"] = frame["Volume"].ffill(limit=ffill_days).fillna(0.0)
    return frame


def build_stock_features(
    frame: pd.DataFrame, periods: Mapping[str, int] | None = None
) -> pd.DataFrame:
    """Price features and indicators for a single stock (8 columns)."""
    periods = dict(periods or INDICATOR_PERIODS)

    # Adjust OHLC with the factor from the Adj Close column (prevents jumps
    # caused by dividends and stock splits).
    factor = frame["Adj Close"] / frame["Close"]
    high = frame["High"] * factor
    low = frame["Low"] * factor
    close = frame["Adj Close"]
    volume = frame["Volume"]

    features = pd.DataFrame(index=frame.index)
    features["close"] = close
    features["volume"] = volume
    features["rsi"] = rsi(close, periods["rsi"])
    features["cci"] = cci(high, low, close, periods["cci"])
    features["cmo"] = cmo(close, periods["cmo"])
    features["mfi"] = mfi(high, low, close, volume, periods["mfi"])
    return features


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Target: log-return `horizon` days ahead."""
    return np.log(close.shift(-horizon) / close)


def build_macro(
    jisdor: pd.DataFrame,
    bi_rate: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    transform: Mapping[str, str],
) -> pd.DataFrame:
    """Align macro data to the exchange calendar by publication date.

    BI-7DRRR is forward-filled without limit because the reference rate stays
    in force until the next decision. The JISDOR exchange rate is converted to
    daily log-returns because its level is non-stationary.
    """
    bi = bi_rate.set_index("date")["rate"].sort_index()
    exchange_rate = jisdor.set_index("date")["rate"].sort_index()

    bi_daily = bi.reindex(calendar, method="ffill")
    exchange_rate_daily = exchange_rate.reindex(calendar, method="ffill")

    if transform.get("jisdor") == "log_return":
        exchange_rate_daily = np.log(exchange_rate_daily / exchange_rate_daily.shift(1))

    return pd.DataFrame({"bi_7drrr": bi_daily, "jisdor": exchange_rate_daily})


def build_panel(
    prices: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    macro: pd.DataFrame,
    horizon: int,
    periods: Mapping[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble the feature + macro + target panel for each stock."""
    panel: dict[str, pd.DataFrame] = {}
    for ticker, raw in prices.items():
        features = build_stock_features(raw, periods)
        features = features.join(macro, how="left")
        features["target"] = forward_log_return(features["close"], horizon)
        panel[ticker] = features
    return panel
