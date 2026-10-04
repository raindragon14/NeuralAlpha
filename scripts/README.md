# scripts

> **Role:** pipeline order and the output of each stage.
> **Audience:** public.
> **Not for:** feature definitions (see `data/README.md`) or decision rationale
> (see `docs/keputusan_desain.md`).

Numbered pipeline, run in order:

| Script | Status | Output |
|---|---|---|
| `01_fetch_data.py` | complete | `data/raw/` - `.JK` prices, `macro/jisdor.csv`, `macro/bi_7drrr.csv`, `benchmark_ihsg.csv`, `metadata.json` |
| `02_build_features.py` | complete | `data/interim/prices_clean/`, `data/processed/features/` (8 features + target) |
| `03_train_predict.py` | complete | `experiments/<run_id>/` - `predictions.csv`, `fold_metrics.csv`, `run_info.json`, `checkpoints/`, `pretrained/` (pretrain mode) |
| `04_optimize.py` | complete | Full grid of 1,728 configurations in `experiments/optimize_20260918_050942/` |
| `05_evaluate.py` | complete | `reports/` - `metrics_table.csv`, `regime_table.csv`, etc. |

Note: `01_fetch_data.py` requires network access to Yahoo Finance and Bank Indonesia. `02_build_features.py` requires no network access.

## 03_train_predict.py

Train a shared CNN-BiLSTM across 43 stocks, then write walk-forward predictions. **New:** `pretrain` mode (self-supervised MAE) and `walk-forward-pretrain` mode (pre-train → fine-tune).

The architecture and split specifications are in `configs/model.yaml` and `configs/split.yaml`; the rationale for each value is in `docs/keputusan_desain.md`.

```
python3 scripts/03_train_predict.py --mode dry-run        # inspect the 65-fold table
python3 scripts/03_train_predict.py --mode smoke          # 2 folds, 1 seed, 3 epochs
python3 scripts/03_train_predict.py --mode calibrate      # measure the smallest/largest fit rate
python3 scripts/03_train_predict.py --mode tune           # grid + design-period sensitivity
python3 scripts/03_train_predict.py --mode pretrain       # NEW: MAE pre-training (self-supervised)
python3 scripts/03_train_predict.py --mode walk-forward-pretrain  # NEW: pre-train + fine-tune
python3 scripts/03_train_predict.py                       # full walk-forward (baseline)
```

Key options: `--seeds 0,1,2,3,4`, `--jobs 4` (parallel; 7 GiB RAM limits this to a maximum of 4), `--resume` (skip completed fold-seed pairs), `--use-tuning <directory>` (freeze the hyperparameters from tuning), `--pretrained-dir <directory>` (source of `encoder_seed{N}.pt` for `walk-forward-pretrain` mode).

### New Mode: Pre-training

```bash
# 1. MAE pre-training (self-supervised, without return labels)
python3 scripts/03_train_predict.py --mode pretrain --epochs 50 --jobs 4 --seeds 0,1,2,3,4
# Output: experiments/pretrain_<timestamp>/pretrained/encoder_seed{0-4}.pt

# 2. Walk-forward with a pre-trained encoder → fine-tune
#    The encoder is read from --pretrained-dir (default <run_dir>/pretrained).
#    Pre-training is NOT run automatically; run step 1 first.
python3 scripts/03_train_predict.py --mode walk-forward-pretrain --jobs 1 --seeds 0,1,2,3,4 \
    --pretrained-dir experiments/pretrain_<timestamp>/pretrained --resume
```

Output contract for stage 4: `predictions.csv` contains one row per (date, stock, seed) with columns `date, ticker, seed, role, pred_scaled, true_scaled, pred_raw, true_raw`. Raw predictions (`pred_raw`) are used for top-k ranking; `role=test` is used for the main evaluation at stage 5.

## 04_optimize.py

Top-k ensemble ranking, then a monthly Mean-Variance backtest with IDX costs. The ranking uses the average `pred_raw` across seeds per (date, stock); the covariance is estimated from L daily log-returns up to the rebalancing date (without look-ahead). The Mean-Variance optimization minimizes variance subject to a target mean return constraint from the pred_ens average (Chaweewanchon & Chaysiri 2022 Section 3.1).

```
python3 scripts/04_optimize.py --predictions experiments/<run_id>/predictions.csv --grid main --seed-mode ensemble
python3 scripts/04_optimize.py --predictions experiments/<run_id>/predictions.csv
```

Default grid (`full`): k {5,7,10,3,15,20}, estimators sample, ridge_epsilon, ledoit_wolf, gmv, L {120,60,252}, maxw {0.35,0.25,0.50,1.0}, ranking ensemble and per seed. Optimizer type: target_return (MV) only. The `--grid main` option runs only k {5,7,10}, L 120, maxw 0.35. The default initial capital is 100 million rupiah.

## 05_evaluate.py

Metrics, significance, and regime robustness from the stage 4 output.

```
python3 scripts/05_evaluate.py --portfolio <opt>/portfolio_returns.csv --baselines <opt>/baseline_returns.csv --weights <opt>/weights.csv
```

Daily risk-free rate = annual BI-7DRRR divided by 252; annualization 252. Turnover is reported as a diagnostic column: the mean |weight difference|/2 per rebalancing. Returns are already net of fees, so no turnover-adjusted Sharpe is computed.
