# About — NeuralAlpha

> **Portfolio showcase page.** Read this as your LinkedIn "about" section.
> **Audience:** HR, recruiters, and hiring managers in Singapore.
> **Not for:** the thesis manuscript (`internal/SKRIPSI.md`) or the decision log (`docs/keputusan_desain.md`).

---

## One-Liner

> I build **neural network systems** that optimize investment portfolios — combining **self-supervised pre-training**, **deep learning sequence modeling**, and **mathematical optimization** under real-world transaction cost constraints.

---

## What I Built

**NeuralAlpha** is a production-grade two-stage portfolio optimization pipeline for Indonesian equities (LQ45 index). The system:

1. **Pre-trains a neural network using self-supervised learning** (Masked Autoencoder) on the 8-feature market panel — no return labels needed, and the pre-training data is capped at the design period so the backtest never sees the future. The encoder learns universal representations of price movements across 43 stocks and transfers into the supervised model unchanged.

2. **Fine-tunes with a CNN-BiLSTM architecture** using walk-forward validation (65 expanding windows, purge + embargo to prevent look-ahead bias). Five-seed ensemble averages predictions to reduce variance.

3. **Selects top-k stocks** based on ensemble predictions, then optimizes portfolio weights using **Mean-Variance optimization** with four covariance estimators (sample, ridge, Ledoit-Wolf, GMV) under realistic Indonesian market constraints (buy 0.19%, sell 0.29%, lot size 100).

4. **Evaluates rigorously** with deflated Sharpe ratio (DSR), probability of backtest overfitting (PBO), Romano-Wolf multiple testing correction, and regime robustness analysis (COVID-19 vs. post-COVID rate hikes).

---

## Technical Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch 2.14, CNN (Conv1d), BiLSTM, Masked Autoencoder |
| **ML Pipeline** | scikit-learn, pandas, numpy |
| **Optimization** | scipy.optimize (SLSQP), custom mean-variance solver |
| **Data** | Yahoo Finance API, Bank Indonesia API (JISDOR, BI-7DRRR) |
| **Infrastructure** | Python 3.11+, Git |
| **Testing** | pytest (30 tests) |

---

## What Makes This Different

| Innovation | Why It Matters |
|------------|----------------|
| **Self-supervised pre-training (MAE)** | Learns price representations without return labels; pre-training data is capped at the design period, so no look-ahead bias from future data |
| **Discriminative learning rates** | Encoder LR 10× smaller than head LR — preserves pre-trained features during fine-tuning |
| **Walk-forward with purge + embargo** | Eliminates look-ahead bias (Lopez de Prado 2018) — industry-standard backtest hygiene |
| **Realistic IDX cost modeling** | Buy/sell fees, lot size (100 shares) — not theoretical, reflects actual Indonesian retail trading |
| **Regime robustness testing** | Validates performance across COVID-19 and rate-hike cycles — not just in-sample |
| **Deflated Sharpe + PBO** | Controls for multiple testing and backtest overfitting — avoids inflated results |

---

## Results Summary

| Dataset | Period | Method | Sharpe (LW-HAC) | Sortino | PBO |
|---------|--------|--------|-----------------|---------|-----|
| LQ45 (43 stocks) | 2018–2025 | Baseline (supervised) | TBD | TBD | TBD |
| LQ45 (43 stocks) | 2018–2025 | **NeuralAlpha (pre-trained)** | TBD | TBD | TBD |
| IHSG | 2018–2025 | Buy-and-hold | TBD | TBD | — |
| LQ45 (43 stocks) | 2018–2025 | 1/N Equal Weight | TBD | TBD | — |

> Full results pending final grid evaluation. See `README.md` for placeholders.

---

## Publications & Academic Context

Built on peer-reviewed foundations:
- **Chaweewanchon & Chaysiri (2022)** — Two-stage ML + Mean-Variance portfolio optimization (IJFS)
- **Huang et al. (2024)** — Two-stage deep learning with GNN + Global Minimum Variance (Mathematics)
- **Lopez de Prado (2018)** — Purged cross-validation methodology (Advances in Financial Machine Learning)
- **TimeMAE / PatchTST** — Self-supervised time series pre-training (Cheng et al. 2023; Nie et al. 2022)

---

## Connect

- **GitHub**: [github.com/raindragon14/NeuralAlpha](https://github.com/raindragon14/NeuralAlpha)
- **Research repo**: [github.com/raindragon14/CNN-BiLSTM-LQ45](https://github.com/raindragon14/CNN-BiLSTM-LQ45)
- **Email**: [your email here]
- **LinkedIn**: [your profile URL]

---

## Keywords for Search

`Deep Learning` · `Neural Networks` · `Portfolio Optimization` · `Machine Learning` · `Time Series Forecasting` · `Self-Supervised Learning` · `PyTorch` · `Quantitative Finance` · `Walk-Forward Validation` · `Mean-Variance Optimization` · `Indonesia Stock Exchange` · `LQ45` · `Backtesting` · `Statistical Significance`