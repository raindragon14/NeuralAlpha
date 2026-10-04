# NeuralAlpha

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

> Two-stage portfolio optimization for LQ45 equities: self-supervised MAE pre-training,
> CNN-BiLSTM walk-forward preselection, and mean-variance optimization with realistic
> IDX transaction costs.

## About

NeuralAlpha is a deep learning research project for portfolio construction on the Indonesian
LQ45 index. It combines three techniques:

- **Self-supervised learning.** A masked autoencoder (MAE) is pre-trained on the same
  8-feature market panel consumed by the supervised model (price, volume, technical
  indicators, BI-7DRRR, JISDOR). Pre-training uses no return labels, so the encoder
  transfers unchanged into the supervised stage. Pre-training data is capped at the design
  period (`pretrain.data_end`) so that nothing from the out-of-sample window leaks in.
- **Deep sequence modeling.** A CNN-BiLSTM architecture predicts 5-day log-returns under
  walk-forward validation (65 expanding windows with purge and embargo). A 5-seed ensemble
  reduces estimation variance.
- **Mathematical optimization.** Ensemble predictions rank stocks and select the top-k;
  mean-variance optimization then allocates weights under Indonesian market constraints
  (buy 0.19%, sell 0.29%, 100-share lots).

Out-of-sample results are assessed with the deflated Sharpe ratio (DSR), the probability of
backtest overfitting (PBO), Romano-Wolf multiple-testing correction, and regime robustness
checks covering COVID-19 and the 2021–2025 rate-hike cycle.

---

## Overview

| Aspect | Details |
|--------|---------|
| Universe | 43/45 LQ45 constituents (Aug 2019–Jan 2020 composition) |
| Period | 2018–2025 (8 years, includes COVID-19 and rate-hike cycles) |
| Architecture | CNN (32→64) + BiLSTM (64×2), MAE self-supervised pre-training on the 8-feature panel (`FEATURE_COLUMNS`) |
| Training | Walk-forward expanding window (65 folds), purge/embargo, 5 seeds, discriminative LR fine-tuning |
| Optimization | Mean-Variance (target-return), 4 covariance estimators, monthly rebalancing (21 days) |
| Costs | Buy 0.19%, sell 0.29%, 100-share lots (IDX retail rules) |
| Evaluation | Sharpe (LW-HAC), Sortino, Calmar, DSR, PBO/CSCV, Romano-Wolf, 3 regime tests |

---

## Method

1. **Data pipeline.** Daily OHLCV from Yahoo Finance (`.JK`), BI-7DRRR (policy rate) and
   JISDOR (USD/IDR) from Bank Indonesia, and the IHSG benchmark.
2. **Feature engineering.** 8 features per stock — price, volume, RSI(14), CCI(20),
   CMO(14), MFI(14), BI-7DRRR (level), JISDOR (log-return). Winsorization and min-max
   scaling are fit on training windows only.
3. **Self-supervised pre-training (MAE).** Masked autoencoder on the same 8-feature panel,
   30% masking, reconstruction loss on masked positions only. This learns market
   representations without return labels. Data is capped at `pretrain.data_end`
   (2019-12-31).
4. **Walk-forward fine-tuning.** The CNN-BiLSTM predicts 5-day log-returns under nested
   walk-forward CV with purge (τ=5) and embargo (w−1=59), an expanding training window,
   and a 5-seed ensemble.
5. **Top-k preselection.** The 5-seed ensemble average ranks stocks; the top-k are
   selected (k = 5, 7, 10, with sensitivity at 3, 15, 20).
6. **Mean-variance optimization.** Minimize variance subject to a target return equal to
   the ensemble mean prediction. Covariance estimators: sample, ridge, Ledoit-Wolf, GMV.
   Maximum weight 35% (sensitivity 25/50/100%).
7. **Evaluation.** Out-of-sample walk-forward metrics, deflated Sharpe (LW-HAC),
   Romano-Wolf multiple-testing correction, probability of backtest overfitting (PBO), and
   regime robustness (COVID 2020; recovery and rate hikes 2021–2025).

---

## Quick Start

### Prerequisites
- Python 3.11+ (tested on 3.13.5)
- PyTorch 2.14+ (CPU)
- 7 GiB RAM minimum (limits parallel jobs to 4)

```bash
git clone https://github.com/raindragon14/NeuralAlpha
cd NeuralAlpha
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pre-commit install
```

A fully pinned environment (Python 3.13, Linux) is in `requirements.lock`.

The stages are also wrapped by a `Makefile` (`make help` lists the targets):

```bash
make reproduce   # data -> features -> train -> optimize -> evaluate
make smoke       # offline end-to-end run on synthetic fixtures
make test        # unit tests with a coverage gate
make lint        # ruff + black --check + mypy
```

### Full Pipeline

```bash
# 1. Fetch raw data (requires internet: Yahoo Finance + Bank Indonesia)
python scripts/01_fetch_data.py
# Output: data/raw/prices/*.csv, data/raw/macro/*.csv, data/raw/benchmark_ihsg.csv

# 2. Build features and targets (offline)
python scripts/02_build_features.py
# Output: data/interim/prices_clean/, data/processed/features/

# 3. Self-supervised MAE pre-training
python scripts/03_train_predict.py --mode pretrain --epochs 50 --jobs 4 --seeds 0,1,2,3,4
# Output: experiments/pretrain_<timestamp>/pretrained/encoder_seed{0-4}.pt

# 4. Walk-forward fine-tuning with the pre-trained encoder (two-stage)
python scripts/03_train_predict.py --mode walk-forward-pretrain --jobs 1 --seeds 0,1,2,3,4 \
    --pretrained-dir experiments/pretrain_<timestamp>/pretrained --resume
# Or the baseline without pre-training:
python scripts/03_train_predict.py --mode walk-forward --jobs 4 --seeds 0,1,2,3,4

# 5. Portfolio optimization (full grid: 1,728 configs)
python scripts/04_optimize.py --predictions experiments/<run_id>/predictions.csv
# Quick grid (12 configs): --grid main --seed-mode ensemble

# 6. Evaluation and regime analysis
python scripts/05_evaluate.py --portfolio <opt>/portfolio_returns.csv --baselines <opt>/baseline_returns.csv --weights <opt>/weights.csv --predictions experiments/<run_id>/predictions.csv
# Output: reports/metrics_table.csv, regime_table.csv, ranker_table.csv, significance.csv, dsr_pbo.json, figures/
```

### Training Modes (`03_train_predict.py`)

```bash
python scripts/03_train_predict.py --mode dry-run        # Inspect the 65-fold table without training
python scripts/03_train_predict.py --mode smoke          # 2 folds, 1 seed, 3 epochs (sanity check)
python scripts/03_train_predict.py --mode calibrate      # Measure sec/epoch on first/last fold
python scripts/03_train_predict.py --mode tune           # Hyperparameter grid + sensitivity on design period
python scripts/03_train_predict.py --mode pretrain       # MAE self-supervised pre-training
python scripts/03_train_predict.py --mode walk-forward-pretrain  # Pre-train then fine-tune (two-stage)
python scripts/03_train_predict.py                       # Walk-forward baseline (supervised only)
```

Key options: `--seeds 0,1,2,3,4`, `--jobs 4` (max 4 with 7 GiB RAM), `--threads`, `--folds`,
`--epochs`, `--resume`, `--use-tuning <dir>`, `--no-checkpoints`.

---

## Repository Structure

```
configs/           YAML configs (model, split, portfolio, data, experiment, universe)
data/              raw/, interim/, processed/ (gitignored — see data/README.md)
docs/              keputusan_desain.md (decision log), references.bib, RESULTS.md
scripts/           Numbered pipeline 01_fetch → 05_evaluate
src/lq45/          Package: data, features, models, portfolio, evaluation, utils
tests/             unit tests + tests/test_pipeline_smoke.py (offline end-to-end)
Makefile           Pipeline entrypoints (make reproduce / smoke / test / lint)
requirements.lock  Fully pinned environment
.github/           CI: lint, type-check and tests on Python 3.11-3.13
```

---

## Current Status (2026-09-22)

| Stage | Status | Experiment ID |
|-------|--------|---------------|
| Data pipeline (01–02) | Complete | — |
| Hyperparameter tuning | Complete | `tune_20260917_062213` (best: batch=64, wd=1e-4, pooling=2, units=64) |
| Walk-forward baseline | Complete | `walk-forward_20260917_081400` (65 folds × 5 seeds) |
| MAE pre-training | Fixed, re-run pending | Masking and scaling bugs fixed; encoders need regenerating |
| Walk-forward + pre-train | Pending | Requires regenerated pre-trained encoders |
| Full grid optimization | Baseline done | Main grid (12): `optimize_20260918_162336`; full grid (1,728): `optimize_20260918_050942` (baseline); pre-train grid queued |
| Evaluation and regime analysis | Pending | Requires full-grid outputs |

Correctness fixes (2026-09-22) changed MAE pre-training (per-sample masking), pre-trained
encoder resolution, MV optimizer determinism, and Romano-Wolf. Further fixes (2026-09-23)
aligned MAE pre-training with the 8-feature panel (encoder transfer), capped pre-training
data at the design period (`pretrain.data_end`), restored the Jan–Apr 2020 OOS window,
corrected the DSR/Romano-Wolf/Sortino formulas, added the Ledoit-Wolf (2008) Sharpe test, a
1-day execution lag, and fee-paying 1/N baselines, and removed `turnover_adjusted_sharpe`.
All earlier `experiments/pretrain_*` outputs are invalid. The figures shown come from
intermediate runs; final results for the journal submission are pending full-grid evaluation
on both the baseline and pre-trained prediction sets.

---

## Key Results

The out-of-sample tables are not yet filled. `docs/RESULTS.md` is the template
they will be written into; `make reproduce` regenerates them from scratch.

---

## Documentation

- `data/README.md` — Data sources, feature definitions, limitations
- `docs/keputusan_desain.md` — Every decision traced to literature and implementation location
- `docs/references.bib` — Machine-readable bibliography (verified against Crossref/arXiv)
- `docs/RESULTS.md` — Results template, filled by `make reproduce`
- `scripts/README.md` — Pipeline contracts and I/O specs
- `CONTRIBUTING.md` — Setup, conventions, pre-push checks
- `CHANGELOG.md` — Released and unreleased changes
- `ABOUT.md` — Project overview (pipeline, contribution, stack)

---

## Citation

If you use this code or its findings, please cite:

```bibtex
@software{neuralalpha,
  author       = {Pandanarang, Reihan},
  title        = {NeuralAlpha: Two-Stage Portfolio Optimization with Self-Supervised MAE Pre-training and CNN-BiLSTM Preselection on LQ45},
  year         = {2026},
  url          = {https://github.com/raindragon14/NeuralAlpha},
  version      = {0.1.0},
  license      = {MIT}
}
```

Full citation metadata: [`CITATION.cff`](CITATION.cff).

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
