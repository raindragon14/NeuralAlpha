# Results

Template for the out-of-sample results. Fill from
`reports/metrics_table.csv`, `reports/regime_table.csv`,
`reports/ranker_table.csv`, `reports/significance.csv` and
`reports/dsr_pbo.json` after running the full grid.

> **Status: pending.** The numbers below are not yet produced.

## Portfolio performance

| Metric | Supervised baseline | NeuralAlpha (pre-trained) | IHSG buy-and-hold | 1/N equal weight |
|---|---|---|---|---|
| Annualized return | — | — | — | — |
| Annualized volatility | — | — | — | — |
| Sharpe ratio | — | — | — | — |
| Sortino ratio | — | — | — | — |
| Calmar ratio | — | — | — | — |
| Maximum drawdown | — | — | — | — |
| Mean turnover per rebalance | — | — | — | — |
| Deflated Sharpe ratio | — | — | — | — |
| Probability of backtest overfitting | — | — | — | — |

## Ranker quality

`reports/ranker_table.csv` — the predictions are consumed as a ranking by the
portfolio stage, so they are also scored as one.

| Metric | Value |
|---|---|
| Mean Rank IC | — |
| Rank IC std | — |
| Rank IC information ratio | — |
| Top-minus-bottom spread (per 5-day horizon) | — |
| Number of dates | — |

## Regime robustness

| Regime | Supervised Sharpe | NeuralAlpha Sharpe | IHSG Sharpe |
|---|---|---|---|
| COVID 2020 (OOS) | — | — | — |
| Recovery and rate hikes 2021-2025 (OOS) | — | — | — |

## Statistical tests

| Comparison | Test | Result |
|---|---|---|
| Best config vs IHSG | Ledoit-Wolf (2008) Sharpe difference | — |
| Best config vs 1/N | Ledoit-Wolf (2008) Sharpe difference | — |
| Main-grid strategies vs 1/N | Romano-Wolf stepdown | — |

## Reproducing these tables

```bash
make reproduce
```

The run id of each stage is recorded in its `run_info.json`.
