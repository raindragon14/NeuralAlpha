# Design Decision Log

> Renamed from `keputusan_desain.md` on 2026-10-04; the content was
> already English, only the filename was not.

> **Role:** the single source of the decision trail -> rationale (paper + location) and
> implementation location.
> **Audience:** public.
> **Not for:** in-depth analysis or rejected alternatives (see
> `internal/First_Principles_Parameter_Analysis.md`) or the thesis manuscript
> (`internal/SKRIPSI.md`).

This file is a trail of every research decision: what was decided, on what
basis, and where it is applied. Its purpose is to make decisions traceable
without relying on memory.

## How to read

- **Rationale** names the paper and the location within the paper (section, table, or
  equation), not just the author's name.
- **Implementation** names the file and key that contain the decision.
- Status: `final` (decided) or `pending` (not yet decided).

---

## Data

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| Period | 2018-01-01 to 2025-12-31 | 8 years; covers COVID 2020 and the 2022-2024 rate hikes | `configs/data.yaml` `period` | final |
| Candidate pool | 45 LQ45 constituents, Aug 2019 - Jan 2020 period | IDX LQ45 Company Profiles (Aug 2019), *Contents* page; Wayback archive | `configs/universe.yaml` | final |
| Reason the pool is frozen | To avoid arbitrary selection and survivorship bias from the end-of-period list | - | `data/README.md` | final |
| Price source | Yahoo Finance `.JK`, `adjusted close` | Secondary source; limitation disclosed | `src/lq45/data/fetch.py` `fetch_equities` | final |
| Exchange-rate source | JISDOR, Bank Indonesia | Official Bank Indonesia reference rate | `src/lq45/data/fetch.py` `fetch_jisdor` | final |
| Interest-rate source | BI-7DRRR, Bank Indonesia | Official BI indicator page | `src/lq45/data/fetch.py` `fetch_bi_rate` | final |
| Benchmark | IHSG (`.JKSE`) | Buy-and-hold comparison | `configs/data.yaml` `sources.benchmark` | final |
| Transaction costs | buy 0.19%, sell 0.29%, lot 100 | Indonesian retail securities terms | `configs/data.yaml` `costs` | final |
| Missing-data handling | forward-fill max. 3 days | Exchange calendars are not synchronized | `configs/data.yaml` `cleaning.forward_fill_max_days` | final |
| Split shape | Initial training 2 years, *expanding* window | Standard walk-forward; an expanding window retains all historical data. Principle, not a citation | `configs/split.yaml` | final |
| Purge & embargo | purge 5 days (=tau), embargo 59 days (=w-1) | Lopez de Prado (2018) Chapter 7 (purged cross-validation): drop training labels that overlap with validation, and leave a gap after the test | `configs/split.yaml` | final |
| Validation/test/step length | 63 / 21 / 21 days | Independent decision: ~1 quarter, 1 month, 1 month; consistent with monthly holding and tau=5 | `configs/split.yaml` | final |
| Macro alignment | Publication date | Principle: prevent look-ahead information leakage | `src/lq45/features/build.py` `build_macro` | final |
| Return type | log-return | Sen & Dutta (2021) compute log returns; additive property over time | `src/lq45/features/build.py` `forward_log_return` | final |

## Features

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| Price representation | `Adj Close` level (not log-return) | Chaweewanchon & Chaysiri (2022) Section 4.1.1 (closing prices); Sebastian & Tantia (2024) data section (6 level features) | `src/lq45/features/build.py` `build_stock_features` | final |
| Base features | `close`, `volume` | Sebastian & Tantia (2024) data section (OHLC + adjusted close + volume); Sen & Dutta (2021) data section (6 features) | `src/lq45/features/build.py` `build_stock_features`; `src/lq45/models/dataset.py` `FEATURE_COLUMNS` | final |
| RSI | period 14 | Espiga-Fernandez et al. (2024) Appendix B.3 | `configs/experiment.yaml` `features.indicators` | final |
| CCI | period 20 | Espiga-Fernandez et al. (2024) Appendix B.4 | `configs/experiment.yaml` `features.indicators` | final |
| CMO | period 14 | Espiga-Fernandez et al. (2024) Appendix B.5 | `configs/experiment.yaml` `features.indicators` | final |
| MFI | period 14 (requires price + volume) | Espiga-Fernandez et al. (2024) Appendix B.6 | `configs/experiment.yaml` `features.indicators` | final |
| Simplification | AROONOSC, Williams %R, STOCHF from Espiga-Fernandez et al. are not included | Deliberately reduced so that the number of features stays at 8 | `src/lq45/models/dataset.py` `FEATURE_COLUMNS` | final |
| Macro: BI-7DRRR | level (%) | Extension; the monthly difference is almost always zero, so it is not informative | `configs/experiment.yaml` `features.macro_transform` | final |
| Macro: JISDOR | daily log-return | No paper uses the macro series as a feature; the JISDOR level is non-stationary, so the log-return is used to make it stationary | `configs/experiment.yaml` `features.macro_transform` | final |
| Number of features | 8 | Literature range from 6 features (Sebastian & Tantia 2024; Sen & Dutta 2021) to 11 features (Espiga-Fernandez et al. 2024) | `src/lq45/models/dataset.py` `FEATURE_COLUMNS` | final |
| Price adjustment | OHLC columns scaled by the `Adj Close / Close` factor | Prevents jumps caused by dividends and stock splits; Sen & Dutta (2021) use `adjusted_close` | `src/lq45/features/build.py` | final |
| Calendar and missing data | Exchange calendar from IHSG; forward-fill max. 3 days | `configs/data.yaml` `cleaning` | `src/lq45/features/build.py` | final |

## Preprocessing

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| Robust transformation | Winsorization thresholds `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` | Sebastian & Tantia (2024) data pre-processing section (outliers via boxplot, capped) | `src/lq45/features/preprocess.py` `RobustPreprocessor.iqr_multiplier` | final |
| Scale | Min-max scaler | Sebastian & Tantia (2024) data pre-processing section | `src/lq45/features/preprocess.py` `RobustPreprocessor.feature_range` | final |
| Transformation scope | Features **and** target | Sebastian & Tantia (2024) handle outliers across the entire dataset before modeling | `scripts/03_train_predict.py` `scale_panel` | final |
| *Fit* point | Only the training data of each walk-forward window | **Principle**, not a citation: preventing look-ahead information leakage | `src/lq45/features/preprocess.py` | final |
| Rejected alternative | Huber location (Chaweewanchon & Chaysiri 2022 Section 3.5.2, `k=1.435`) | The paper does not explain standardization before Huber, so `k=1.435` on raw prices effectively becomes the median; hard to replicate exactly | `docs/DECISIONS.md` | final |

## Target and Training

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| Target | 5-day-ahead log-return | Consistent with the horizon `tau=5` and monthly holding | `configs/model.yaml` `horizon_days` | final |
| Loss function | MSE | Chaweewanchon & Chaysiri (2022) hyperparameter selection section ("Mean Squared Error (MSE) was used as the loss function"); Kim et al. (2025) ("standard loss function (e.g., mean squared error)") | `src/lq45/models/training.py` `_run_epochs` | final |
| Rejected loss alternative | Huber loss | Not used by any paper in the folder; the robustness requirement is already handled by winsorization (Sebastian & Tantia 2024); MV requires a mean estimate, not a median | `docs/DECISIONS.md` | final |
| Look-back window | `w=60` | Independent decision (common practice, bias-variance trade-off); tested in the window sensitivity analysis | `configs/model.yaml` `lookback_days` | final |
| Prediction horizon | `tau=5` | Independent decision (consistent with monthly holding); tested in the window sensitivity analysis | `configs/model.yaml` | final |
| Covariance estimation window | `L=120` | No paper sets it; sensitivity tested at the portfolio stage (L tested at 60, 120, 252) | `configs/portfolio.yaml` | final |
| Seed averaging | 0, 1, 2, 3, 4 | Dampens training variance (independent decision) | `scripts/03_train_predict.py` `DEFAULT_SEEDS` | final |
| Learning rate | 0.0001 | Chaweewanchon & Chaysiri (2022) Section 4.1.3; that paper also cites Hastie et al. (2017) that lr < 0.01 | `configs/model.yaml` | final |
| Optimizer | Adam | Chaweewanchon & Chaysiri (2022) Section 4.1.3; Sebastian & Tantia (2024) | `src/lq45/models/training.py` `train_model` | final |
| Number of epochs | 100 | Chaweewanchon & Chaysiri (2022) Section 4.1.3 (training stopped at 100-120 epochs); Sebastian & Tantia (2024) | `configs/model.yaml` | final |
| Early stopping | patience 8 | Sebastian & Tantia (2024) (stop after 8 epochs without improvement) | `configs/model.yaml` `early_stopping.patience` | final |
| Dropout | uniform 0.2 (CNN, LSTM, Dense) | Sebastian & Tantia (2024) (dropout 0.2); the previous figure of 0.3 was unfounded | `configs/model.yaml` | final |
| Activation function | ReLU | Espiga-Fernandez et al. (2024) Table 3 (ReLU on both convolution layers and the Linear layer) | `configs/model.yaml` | final |
| CNN filters | Two layers 32 -> 64 | Espiga-Fernandez et al. (2024) Table 3 uses exactly this pattern | `configs/model.yaml` | final |
| CNN kernel | 3 | Chaweewanchon & Chaysiri (2022) 3x3 | `configs/model.yaml` | final |
| BiLSTM units | 64 | No paper sets it; tuned on the design period. The grid deliberately covers the values of Chaweewanchon & Chaysiri (2022) = 128 and Sebastian & Tantia (2024) = 32 | `configs/model.yaml` | final |
| BiLSTM layers | 2 | Graves, Mohamed & Hinton (2013) introduced stacked RNNs; Chaweewanchon & Chaysiri (2022) and Sebastian & Tantia (2024) use two layers | `configs/model.yaml` | final |
| Weight decay | 0 | No paper sets it; tuned on the design period (0 vs 0.0001) | `configs/model.yaml` | final |
| Pooling | MaxPool 2 | Papers mention pooling without a uniform value; tuned on the design period (2 vs 3) | `configs/model.yaml` | final |
| BatchNorm | Yes | Chaweewanchon & Chaysiri (2022) use BatchNorm | `configs/model.yaml` | final |
| Device | `auto` (CUDA if available, otherwise CPU) | The official CPU path preserves the resource-efficiency and reproducibility claims; GPU is only an additional experiment | `configs/model.yaml` | final |
| Batch size | 32 | No paper sets it; tuned on the design period (32 vs 64). Huang et al. (2024) use the number of stocks as the batch | `configs/model.yaml` | final |
| Model scope | A single model shared across all stocks | Espiga-Fernandez et al. (2024) input `lookback x instruments x features`; Huang et al. (2024) input `32 x 15`; this also makes training feasible on CPU | `configs/model.yaml` | final |
| Weak-item procedure | Tune the BiLSTM units, weight decay, pooling, and batch size on the design period (train 2018, validate 2019), then freeze; report the 24-combination grid as an appendix | Turns assumptions into an auditable selection procedure; prevents leakage because it does not touch the OOS data. The unit grid deliberately covers the paper values (32 = Sebastian & Tantia 2024, 128 = Chaweewanchon & Chaysiri 2022) | `configs/model.yaml` | final |
| Window sensitivity | `w` {30, 60, 120} and `tau` {1, 5, 21} tuned on the design period | Makes `w` and `tau` test results rather than assumptions; results reported as an appendix | `configs/model.yaml` `sensitivity` | final |
| Rejected architecture alternatives | Attention-based hybrid models (CNN-BiLSTM-Attention/ECA, BiLSTM-Transformer) | They are the 2024-2025 literature direction, but require more data than 43 stocks x 8 years; the scope of this research is a replication of CNN-BiLSTM (Chaweewanchon & Chaysiri 2022). Noted as future work | `internal/SKRIPSI.md` Section 4.9 | final |

## Portfolio and Evaluation

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| Final portfolio size | `k` = 5, 7, 10 | Research question RQ2 | `configs/portfolio.yaml` | final |
| Weight cap | `max_weight` = 35% | No paper sets it; sensitivity tested at the portfolio stage (25%, 35%, 50%, 100%) | `configs/portfolio.yaml` | final |
| Covariance estimators | sample, ridge, Ledoit-Wolf, GMV (comparison, not assumption) | DeMiguel et al. (2009); Ledoit & Wolf (2004) | `configs/portfolio.yaml` | final |
| Rebalancing | monthly (21 days) | Espiga-Fernandez et al. (2024) (periodic rebalancing is more cost-efficient); Huang et al. (2024) | `configs/portfolio.yaml` | final |
| Metrics | Sharpe, Sortino (downside deviation over all periods), Calmar, MDD, turnover, DSR, PBO | Bailey & Lopez de Prado (2014); Bailey et al. (2016); Sortino downside-deviation convention | `src/lq45/evaluation/metrics.py` `summarize_series`; `src/lq45/evaluation/dsr_pbo.py` | final |
| Significance tests | Ledoit-Wolf (2008) Sharpe-difference test (delta method over NW-HAC moments) with the HAC mean-difference test as supplement; Romano-Wolf for multiple testing | Ledoit & Wolf (2008) robustify the Jobson & Korkie (1981) test against non-normality and time-series dependence; Romano & Wolf (2005) for data snooping | `src/lq45/evaluation/significance.py` | final |
| Regimes | COVID 2020; recovery/rate hikes 2021-2025 (both OOS) | Huang et al. (2024) (COVID robustness test). Caveat: the OOS evaluation includes fold-0's validation window (Jan-Mar 2020, the crash); those days double as fold-0 early-stopping validation - disclosed here | `configs/experiment.yaml` | final |
| Risk-free rate | daily BI-7DRRR (= annual / 252) | Bank Indonesia (official source); simple conversion by 252 days | `src/lq45/evaluation/metrics.py` `daily_risk_free_rate` | final |
| Annualization factor | 252 | Standard convention. IDX empirically ~242 days/year; tested as a sensitivity | `configs/experiment.yaml` `evaluation.annualization` | final |
| MV constraints | long-only, weights sum = 1 | Markowitz (1952); IDX retail trading does not allow shorting | `src/lq45/portfolio/optimize.py` `minimum_variance_weights`; `configs/portfolio.yaml` `constraints.max_weight` | final |
| Covariance regularization | ridge `epsilon` = 1e-4 | Ledoit & Wolf (2004) shrinkage for large-dimensional covariance; ridge stabilizes the diagonal | `configs/portfolio.yaml` `covariance.ridge_epsilon` | final |
| Turnover penalty | 0 | Transaction costs are simulated explicitly (buy 0.19%, sell 0.29%, lot 100), not through a penalty | `src/lq45/portfolio/optimize.py` `mean_variance_target_return_weights` (objective carries no turnover term) | final |
| Comparison models | classical MV, Ridge, Random Forest, Gradient Boosting, CNN-BiLSTM; 1/N and IHSG baselines | Chaweewanchon & Chaysiri (2022) compare LSTM/BiLSTM/CNN-BiLSTM/MV; DeMiguel et al. (2009) for 1/N; Anuno & Madaleno (2024) for classical MVO | — (planned: `scripts/06_ablation.py`) | pending |
| `k` grid | 5, 7, 10 (RQ2) | Chaweewanchon & Chaysiri (2022) test N = 5-10; Paiva et al. (2019) = 7; Wang et al. (2020) = 10 | `configs/portfolio.yaml` `selection.k_values` | final |
| Sharpe test | Ledoit-Wolf HAC | Ledoit & Wolf (2008) | `src/lq45/evaluation/significance.py` `sharpe_difference_test_lw` | final |
| Multiple testing | Romano-Wolf stepdown | Romano & Wolf (2005); Bailey et al. (2016) for backtest overfitting risk | `src/lq45/evaluation/significance.py` `romano_wolf_stepdown` | final |
| Sensitivity of independent figures | `w` {30,60,120}, `tau` {1,5,21}, `L` {60,120,252}, `maxw` {25,35,50,100}%, `k` {3,15,20} | Turns figures without a paper source into test results; reported as an appendix | `configs/model.yaml` `sensitivity`; `configs/portfolio.yaml` | final |
| Rejected validation alternative | Combinatorial Purged Cross-Validation (CPCV) | It is the 2024-2025 standard practice, but its method family is already represented by PBO/CSCV (Bailey et al. 2016); nested walk-forward with purge/embargo (Lopez de Prado 2018) is already used | `internal/SKRIPSI.md` Section 4.9 | final |
| Mean-Variance formulation | Target-return constrained: min w'Σw s.t. w'μ=γ, Σw=1, 0≤w≤maxw | **Chaweewanchon & Chaysiri (2022) Section 3.1 Equations (1)-(4)** — an explicit MV formulation with expected returns from ML predictions; Wang et al. (2020) LSTM+MV use predictions as expected returns | `src/lq45/portfolio/optimize.py` `mean_variance_target_return_weights` | final |
| Expected return (μ) for MV | Ensemble mean of `pred_ens` (5 seeds) from the selected top-k stocks | Chaweewanchon & Chaysiri (2022) "predicted results are integrated into the MV model"; Wang et al. (2020) LSTM+MV use predictions as expected returns | `src/lq45/portfolio/backtest.py` `_target_return_from_predictions` | final |
| Target return (γ) | Mean pred_ens of the k selected stocks on the rebalancing date | Natural choice, no extra hyperparameter; consistent with Ei in the Chaweewanchon & Chaysiri (2022) equation | `src/lq45/portfolio/backtest.py` `_target_return_from_predictions` | final |
| Optimizer type grid | `target_return` (MV) only; GMV removed from the grid | Huang et al. (2024) Table 1 Panel B compares AGC-CNN+GMV vs AGC-CNN+MaxSR vs AGC-CNN+1/N; MV was chosen because it uses ML predictions as expected returns | `configs/portfolio.yaml` `optimization.types` | final |
| Risk aversion λ | Not used (target-return formulation) | The Chaweewanchon & Chaysiri (2022) Eq 1-4 formulation does not use λ | - | final |
| Optimizer type in run_info | Additional column in the output for auditing | Design-decision traceability | `scripts/04_optimize.py` `run_info.json` | final |

---

## Stage 3 Implementation

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| **Self-Supervised Pre-training** | Masked Autoencoder (MAE) on the 8 `FEATURE_COLUMNS`, mask_ratio=0.3, reconstruct all input channels at masked positions | PatchTST (Nie et al. 2022), TimeMAE (Cheng et al. 2023), MTSMAE (Tang & Zhang 2022): MAE pre-training on time series improves downstream forecasting; the input must equal the supervised panel for the encoder to transfer (decision 2026-09-23, revises the raw-OHLCV variant of Kang 2025) | `configs/model.yaml` `pretrain`; `src/lq45/models/pretrain.py`; `scripts/03_train_predict.py` `pretrain` mode | final |
| **Fine-tuning with Discriminative LR** | Head LR=1e-4, Encoder LR=1e-5 (10x smaller), freeze_encoder=false (full unfreeze) | Standard transfer-learning practice; the pre-trained encoder has already learned universal representations, so only the head needs fast adaptation | `configs/model.yaml` `pretrain.fine_tune`; `src/lq45/models/training.py` `fine_tune_model`; `scripts/03_train_predict.py` `walk-forward-pretrain` mode | final |
| **Pre-train Input** | 8 `FEATURE_COLUMNS` (close, volume, rsi, cci, cmo, mfi, bi_7drrr, jisdor) | Channel alignment with the supervised model is required for encoder transfer (the earlier 7-channel raw-OHLCV variant failed to load: conv shapes (32,7,3) vs (32,8,3)) | `src/lq45/models/dataset.py` `FEATURE_COLUMNS` | final |
| **Pre-train Target** | Reconstruct all 8 input channels at masked positions | Matches MTSMAE/PatchTST (masked-only reconstruction over the full input); consistent with the aligned input panel | `src/lq45/models/decoder.py` `n_channels=8`; `src/lq45/models/pretrain.py` `mae_loss` | final |
| **Pre-train Split** | Train 90% / Val 10% of the pre-training windows per stock, on data up to `pretrain.data_end` (2019-12-31) | Unlabeled ≠ no leakage in a backtest: capping pre-training at the design period keeps the encoder blind to the OOS window (decision 2026-09-23, revises the earlier all-data claim) | `src/lq45/models/dataset.py` `pretrain_cutoff_index`, `build_pretrain_windows`; `scripts/03_train_predict.py` `run_pretrain` | final |

## Stage 3 Implementation (continued)

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| CNN choices not fixed by the papers | Padding `same`, order Conv -> BatchNorm -> ReLU, default PyTorch initialization | The papers do not pin these details down; recorded explicitly so they can be audited | `src/lq45/models/cnn_bilstm.py` | final |
| Preprocessing fit point | Fit on the training date rows of all stocks pooled together (not on window elements); the transform works per element, so the result is identical | Consistent with the preprocessing fit-point principle (training data only) and the model shared across all stocks | `scripts/03_train_predict.py` `scale_panel` | final |
| Target cross-check | The target is recomputed from `close`; for tau=5 it is matched against the `target` column of the CSV (`allclose`) | Prevents an undetected pipeline shift at stage 2 | `src/lq45/models/dataset.py` `load_panel` | final |
| Purge and embargo follow the selected w/tau | Purge = tau, embargo = w-1 (not the default values 5/59 if the tuning results differ) | Consistent with the definition in Lopez de Prado (2018) Chapter 7 | `scripts/03_train_predict.py` call to `make_folds` | final |
| Prediction deduplication | One row per (date, stock, seed) from the smallest fold that contains that date | The validation of the next fold overlaps with the test of the previous fold (step 21 < test 21); the smallest fold = first occurrence of the date in evaluation | `scripts/03_train_predict.py` `merge_predictions` | final |
| Opening OOS validation predictions | The first 63 validation days are retained (before the first test) for stage 4 with no date gaps | Stage 4 requires a prediction on every OOS rebalancing date | `scripts/03_train_predict.py` `merge_predictions` | final |
| Determinism | `build_model` plants the seed before initialization; two runs with the same seed are bitwise identical (verified by a test) | Reproducibility; the cross-seed spread is reported (Reimers & Gurevych 2017) | `src/lq45/models/training.py`; `tests/test_determinism.py` | final |
| Resumption and parallelism | Resumption of interrupted execution per (fold, seed); atomic file writes; process parallelism with a thread per worker | Long runs can be resumed and use 8 cores without breaking determinism | `scripts/03_train_predict.py` | final |
| Data missing due to suspension | Windows with NaN are dropped; stocks without a prediction on that date are excluded from the ranking | WIKA was suspended from 2024-01-09 to 2024-04-05 and from 2025-03-07 to the end of the period; SRIL and WSKT are unavailable on Yahoo Finance | `src/lq45/models/dataset.py` `build_windows`; `scripts/03_train_predict.py` `load_panel_data` | final |

---

## Ranker Evaluation Gate

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| Model evaluated as a ranker | Rank IC and top-minus-bottom spread, not MSE/R2 | The stage 4 portfolio decision uses only the ordering; scale shrinkage preserves the ordering but breaks MSE | `scripts/05_evaluate.py` | final |
| Cross-seed ensemble ranking | Average of pred_raw across five seeds; the spread is reported | Dampens training variance (Reimers & Gurevych 2017; Bouthillier et al. 2021) | `src/lq45/portfolio/ranking.py` `ensemble_predictions` | final |
| All out-of-sample rows are evaluated | Stage 4 keeps every deduplicated prediction row (the `role` column is audit-only) | Restores fold-0's validation window (Jan-Mar 2020) so the COVID crash is inside the evaluation; caveat: those days double as fold-0 early-stopping validation (disclosed in the Regimes row) | `scripts/04_optimize.py` `prediction_frame` | final |

---

## Stage 4-5 Implementation

| Decision | Value | Rationale | Implementation | Status |
|---|---|---|---|---|
| Covariance from data up to the rebalancing date | Window `(d-L, d]` of Adj Close log-returns | Principle: prevent look-ahead information leakage | `src/lq45/portfolio/covariance.py` | final |
| gmv label uses the sample covariance | Same as sample under the same constraints | Huang et al. (2024) use Global Minimum Variance at the second stage; kept as a named comparison | `src/lq45/portfolio/covariance.py` | final |
| Minimum-variance optimization SLSQP | Long-only, sum to one, weight cap; failure means equal weighting | Markowitz (1952); IDX retail constraints (long-only, sum=1) | `src/lq45/portfolio/optimize.py` | final |
| Mean-variance target-return optimization | min w'Σw s.t. w'μ=γ; fallback to GMV if infeasible | Chaweewanchon & Chaysiri (2022) Section 3.1 Equations (1)-(4) | `src/lq45/portfolio/optimize.py` `mean_variance_target_return_weights` | final |
| Expected returns passed to optimizer | `pred_ens` series from the ensemble predictions per rebalancing date | Wang et al. (2020); Chaweewanchon & Chaysiri (2022) | `src/lq45/portfolio/backtest.py` | final |
| Lot rounding before fees | Target shares are rounded down to a multiple of 100; the fee is computed from the change in position | Indonesian retail securities terms (buy 0.19%, sell 0.29%, lot 100) | `src/lq45/portfolio/costs.py` | final |
| Initial capital 100 million rupiah | Default value of `--capital`; can be changed | Approximates the retail scale so the lot effect is realistic; not a citation | `scripts/04_optimize.py` | final |
| Rebalancing schedule from the test calendar | Every 21 test prediction dates | Consistent with the 21-day test window at stage 3 | `src/lq45/portfolio/backtest.py` | final |
| Turnover diagnostic | Turnover = mean \|weight difference\|/2 per rebalancing, reported as-is | Returns are already net of fees; the earlier `turnover_adjusted_sharpe = Sharpe x (1 - turnover)` was removed as unfounded | `scripts/05_evaluate.py` | final |
| Execution lag | Trades execute 1 trading day after the signal date | Principle: never trade on the bar the decision was made from; Kim et al. (2025) use a 2-day lag | `configs/portfolio.yaml` `execution_lag_days`; `src/lq45/portfolio/backtest.py` | final |
| 1/N baseline parity | 1/N rebalances every 21 days and pays the same fees and lot rules as the strategies | DeMiguel et al. (2009) use a periodically rebalanced 1/N; a free daily-rebalanced 1/N is not comparable | `scripts/04_optimize.py` `baselines` | final |
| DSR trial correction | n_trials = number of metric rows; SR0 scaled by the variance of the trials' Sharpe ratios (per-period units) | Bailey & Lopez de Prado (2014) Eq. (1)-(2), reference `getExpMaxSR(mu, sigma, numTrials)`: sigma is the dispersion of the trials' SRs | `src/lq45/evaluation/dsr_pbo.py`; `scripts/05_evaluate.py` | final |

---

## Article references

Machine-readable bibliography: `docs/references.bib`.

All entries marked **(OA)** are open access and can be downloaded directly.

- Anuno, F., Madaleno, M., & Vieira, E. (2024). Testing of Portfolio Optimization by Timor-Leste Portfolio Investment Strategy on the Stock Market. *JRFM*, 17(2), 78. https://doi.org/10.3390/jrfm17020078 **(OA)**
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*. PDF: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf **(OA)**
- Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). The Probability of Backtest Overfitting. *Journal of Computational Finance*. PDF: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf **(OA)**
- Bouthillier, X., Delaunay, P., Bronzi, M., et al. (2021). Accounting for Variance in Machine Learning Benchmarks. *MLSys*. https://arxiv.org/abs/2103.03098 **(OA)**
- Chaweewanchon, A., & Chaysiri, R. (2022). Markowitz Mean-Variance Portfolio Optimization with Predictive Stock Selection Using Machine Learning. *IJFS*, 10(3), 64. https://doi.org/10.3390/ijfs10030064 **(OA)**
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*. Working version: https://users.nber.org/~confer/2006/si2006/ap/uppal.pdf **(OA)**
- Espiga-Fernandez, F., Garcia-Sanchez, A., & Ordieres-Mere, J. (2024). A Systematic Approach to Portfolio Optimization: A Comparative Study of Reinforcement Learning Agents, Market Signals, and Investment Horizons. *Algorithms*, 17(12), 570. https://doi.org/10.3390/a17120570 **(OA)**
- Graves, A., Mohamed, A.-R., & Hinton, G. (2013). Speech Recognition with Deep Recurrent Neural Networks. *ICASSP*, 6645-6649. PDF: https://www.cs.toronto.edu/~graves/icassp_2013.pdf **(OA)**
- Huang, S., Cao, L., Sun, R., Ma, T., & Liu, S. (2024). Enhancing Portfolio Optimization: A Two-Stage Approach with Deep Learning and Portfolio Optimization. *Mathematics*, 12(21), 3376. https://doi.org/10.3390/math12213376 **(OA)**
- Kim, J., Choi, S., Lee, Y., Kim, Y., Choi, Y., & Lee, Y. (2025). Decision by Supervised Learning with Deep Ensembles: A Practical Framework for Robust Portfolio Optimization. arXiv:2503.13544 (CIKM 2025 FinAI Workshop). https://arxiv.org/abs/2503.13544 **(OA)**
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. Working version: http://www.ledoit.net/honey.pdf **(OA)**
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850-859. PDF: http://www.ledoit.net/jef_2008pdf.pdf **(OA)**
- Malhotra, D. K., Mooney, T., Poteau, R., & Russel, P. (2023). Assessing the Performance and Risk-Adjusted Returns of Financial Mutual Funds. *IJFS*, 11(4), 136. https://doi.org/10.3390/ijfs11040136 **(OA)**
- Reimers, N., & Gurevych, I. (2017). Reporting Score Distributions Makes a Difference. *EMNLP*. https://aclanthology.org/D17-1035 **(OA)**
- Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica*, 73(4), 1237-1282. Working version available at the Cowles Foundation. **(OA working version)**
- Sebastian, T. A., & Tantia, R. (2024). Deep Learning Stock Price Prediction and Portfolio Optimization. *IJACSA*. https://thesai.org/Downloads/Volume15No9/Paper_95-Deep_Learning_for_Stock_Price_Prediction.pdf **(OA)**
- Sen, J., Dutta, A., & Mehtab, S. (2021). Stock Portfolio Optimization Using a Deep Learning LSTM Model. arXiv:2111.04709. https://arxiv.org/abs/2111.04709 **(OA)**
- Wang, W., Li, W., Zhang, N., & Liu, K. (2020). Portfolio formation with preselection using deep learning from long-term financial data. *Expert Systems with Applications*. Repository PDF: https://centaur.reading.ac.uk/86775/3/portfolio%20formation_revised2_20191010.pdf **(OA)**
- Wang, X., & Liu, X. (2025). Risk-Sensitive Deep Reinforcement Learning for Portfolio Optimization. *JRFM* (MDPI). **(OA)**

**New for Pre-training (OA):**
- Kang, S. (2025). Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets. arXiv:2504.02249. https://arxiv.org/abs/2504.02249 **(OA)**
- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST). arXiv:2211.14730. https://arxiv.org/abs/2211.14730 **(OA)**
- Cheng, M., Tao, X., Liu, Z., Liu, Q., Zhang, H., Zhang, R., & Chen, E. (2023). TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders. arXiv:2303.00320. https://arxiv.org/abs/2303.00320 **(OA)**
- Tang, P., & Zhang, X. (2022). MTSMAE: Masked Autoencoders for Multivariate Time-Series Forecasting. arXiv:2210.02199. https://arxiv.org/abs/2210.02199 **(OA)**

**Not open access** (books and classic papers; cited as rationale, not for download):

- Jobson, J. D., & Korkie, B. M. (1981). Performance Hypothesis Testing with the Sharpe and Treynor Measures. *Journal of Finance*, 36(4), 889-908. The original test refined by Ledoit & Wolf (2008).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Basis for purging and embargo (Chapter 7).
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Paiva, F. D., et al. (2019). Decision-making for financial trading. *Expert Systems with Applications*. Cited via Chaweewanchon & Chaysiri (2022).

Note: PDF copies of the articles are in `Literature/` (internal, not
published due to copyright). This file contains only citations, not content.
