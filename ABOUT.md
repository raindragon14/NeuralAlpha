# About — NeuralAlpha

NeuralAlpha is a research pipeline for portfolio construction on the Indonesian
LQ45 index. It combines self-supervised pre-training, CNN-BiLSTM sequence
modelling and mean-variance optimization under Indonesian retail transaction
costs, and evaluates the result with the validation battery used in the
backtest-overfitting literature.

This file is the project overview. For the method, see the `README.md`; for the
decision trail, see `docs/keputusan_desain.md`.

## Pipeline

1. **Data.** Daily OHLCV from Yahoo Finance (`.JK`), BI-7DRRR and JISDOR from
   Bank Indonesia, IHSG as the benchmark. Sources and limitations: `data/README.md`.
2. **Features.** Eight per stock: adjusted close, volume, RSI(14), CCI(20),
   CMO(14), MFI(14), BI-7DRRR level, JISDOR log-return.
3. **Pre-training.** A masked autoencoder is pre-trained on the same 8-feature
   panel with no return labels. Pre-training data is capped at the design
   period, so the encoder never sees the out-of-sample window.
4. **Fine-tuning.** A CNN (32→64) plus a two-layer BiLSTM predicts 5-day
   log-returns under nested walk-forward validation with purge and embargo; a
   five-seed ensemble averages the predictions.
5. **Portfolio.** Top-k preselection from the ensemble, then mean-variance
   optimization under long-only, sum-to-one and a weight cap, with buy/sell
   fees and 100-share lots applied at execution.
6. **Evaluation.** Sharpe (Ledoit-Wolf HAC), Sortino, Calmar, maximum drawdown,
   turnover, Rank IC, deflated Sharpe, PBO/CSCV, Romano-Wolf, and regime splits.

## Contribution

The contribution is empirical rather than architectural:

- A cost-aware, multiple-testing-corrected comparison of self-supervised
  pre-training against supervised-only training on an emerging market, using
  identical walk-forward splits for both arms.
- The predictions are scored as a ranker (Rank IC, top-minus-bottom spread)
  as well as through portfolio metrics, because the portfolio stage consumes
  only the ordering.
- Every research decision is traced to its literature basis and the code that
  implements it, in `docs/keputusan_desain.md`.

## Status

The pipeline is implemented and unit-tested. The out-of-sample result tables
are not yet filled; see `docs/RESULTS.md`.

## Technical stack

| Category | Tools |
|---|---|
| Deep learning | PyTorch (CNN, BiLSTM, masked autoencoder) |
| ML pipeline | scikit-learn, pandas, numpy |
| Optimization | `scipy.optimize` (SLSQP), custom mean-variance solver |
| Data | Yahoo Finance, Bank Indonesia (JISDOR, BI-7DRRR) |
| Quality | pytest, ruff, black, mypy, GitHub Actions |

## Reproducing

```bash
make reproduce   # data -> features -> train -> optimize -> evaluate
make smoke       # offline end-to-end run on synthetic fixtures
make test        # unit tests with a coverage gate
```

A fully pinned environment is in `requirements.lock`.

## Links

- **Code**: [github.com/raindragon14/NeuralAlpha](https://github.com/raindragon14/NeuralAlpha)

## Citing

See [`CITATION.cff`](CITATION.cff).
