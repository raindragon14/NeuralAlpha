# Data

> **Role:** data sources, feature definitions, and limitations.
> **Audience:** public.
> **Not for:** rationale for decisions (see `docs/DECISIONS.md`) or the
> script order (see `scripts/README.md`).

The research data is not included in the repository (file size and source terms).

## Structure

- `raw/`: raw download output, unmodified
  - `prices/*.csv`: daily OHLCV per stock (columns `Close` and `Adj Close`)
  - `macro/jisdor.csv`: official USD/IDR reference exchange rate (BI)
  - `macro/bi_7drrr.csv`: history of BI policy rate decisions
  - `metadata.json`: download time, ticker list, row count, tickers without data
- `interim/`: cleaning output (exchange calendar, forward-fill max. 3 days, log-returns)
- `processed/features/`: 8 features + target ready for the model (see the Features and target section)

## Sources and justification

| Data | Source | Nature | Notes |
|---|---|---|---|
| Stock prices | Yahoo Finance (`.JK`), `adjusted close` | Secondary | Adjusted for splits and dividends; disclosed as a limitation |
| USD/IDR exchange rate | Bank Indonesia, JISDOR | Primary | Official reference rate; downloaded from `bi.go.id` |
| Policy rate | Bank Indonesia, BI-7DRRR | Primary | Date of decision publication, not effective date |
| Benchmark | IHSG (`.JKSE`) | Secondary | For the buy-and-hold comparison |

Accountability notes:

- The two macro variables come from **Bank Indonesia** (a primary source), not a
  third-party aggregator.
- Stock prices come from Yahoo Finance. This is not a primary source,
  so the price adjustment method (splits/dividends) is not fully
  transparent. This limitation must be stated in the report; cross-validation
  against official IDX sources is recommended before the final results are reported.
- Macro dates use the **publication date** to prevent look-ahead
  information leakage.
- Daily JISDOR values are used on the same date (no additional lag).
  JISDOR is published in the afternoon, so using it immediately is
  contemporaneous; the impact is limited because it is only one of eight features and
  is noted here as a limitation.

## Candidate pool

The pool is read from `configs/universe.yaml`:

- **Composition:** the 45 LQ45 constituents in effect during the period August 2019 -
  January 2020.
- **Source:** IDX LQ45 Company Profiles, August 2019 (*Contents* page).
  The archived document can be verified on the Wayback Machine.
- **Rationale:** this composition was in effect immediately before the out-of-sample period
  began (January 2020), so it does not use future information and does not
  introduce *survivorship* bias from the end-of-period list. The pool was then
  frozen for the entire research period.
- **Availability:** 43 of 45 tickers are available on Yahoo Finance. `SRIL.JK` and
  `WSKT.JK` are unavailable due to suspension/delisting.

Limitations that must be stated:

- The pool is frozen, not *point-in-time*. Stocks that entered or left LQ45
  after January 2020 are not treated dynamically.
- The two suspended stocks cannot be included due to limitations
  in the price data.

## Features and target

Eight features per stock. Specification: `configs/experiment.yaml`; decision
trail: `docs/DECISIONS.md`.

| Feature | Value | Source |
|---|---|---|
| `close` | `Adj Close` level | Sebastian & Tantia (2024); Sen & Dutta (2021) |
| `volume` | Daily volume | Sebastian & Tantia (2024); Sen & Dutta (2021) |
| `rsi` | RSI period 14 | Espiga-Fernandez et al. (2024) Appendix B.3 |
| `cci` | CCI period 20 | Espiga-Fernandez et al. (2024) Appendix B.4 |
| `cmo` | CMO period 14 | Espiga-Fernandez et al. (2024) Appendix B.5 |
| `mfi` | MFI period 14 | Espiga-Fernandez et al. (2024) Appendix B.6 |
| `bi_7drrr` | BI-7DRRR level | Extension (gap in Chaweewanchon & Chaysiri 2022) |
| `jisdor` | Log-return USD/IDR | Extension (ibid.) |

- **Target**: 5-day-ahead log-return.
- **Preprocessing**: winsorization `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` then min-max,
  applied to the features and the target, *fit* **only on the training data** of each
  walk-forward window (preventing information leakage).
- AROONOSC, Williams %R, and STOCHF from Espiga-Fernandez et al. (2024) are not
  included so that the number of features stays at 8. This simplification is intentional.

## Transaction costs

- Buy fee 0.19%, sell fee 0.29%, lot 100 shares (retail securities terms).
- Monthly rebalancing (21 trading days).
