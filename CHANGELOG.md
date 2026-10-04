# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Offline end-to-end pipeline smoke test on synthetic fixtures, so the
  pipeline is verifiable without live Yahoo Finance / Bank Indonesia access.
- `rank_ic_and_spread` and `05_evaluate.py --predictions`, reporting the Rank
  IC and the top-minus-bottom spread of the stage-3 predictions.
- `docs/references.bib`, with every entry verified against Crossref or the
  arXiv API.

### Changed
- Merged the pre-trained and supervised training paths into `run_fit` and
  `run_walkforward`; the shared prediction and metrics tail now lives once.
- Extracted a single `_run_epochs` loop shared by `train_model` and
  `fine_tune_model`.
- Re-pointed the design decision log's implementation references at the code
  that actually holds each value.
- Validated inputs at the trust boundaries: configuration files must be YAML
  mappings, and CSV readers name the missing column instead of failing later.

### Fixed
- Five wrong titles/authors in the decision log's reference list and three
  wrong author names in the generated bibliography, found by verifying every
  entry against Crossref/arXiv.
- `pbo_cscv` no longer accepts an unused `seed` argument.
- The data-fetch retry path logs the failure reason instead of swallowing
  every exception silently.

### Removed
- Unused helpers, config keys that no code read, and the unfounded
  `turnover_adjusted_sharpe` metric.

## [0.1.0] - 2026-09-20

### Added
- Initial pipeline: data fetch, feature engineering, MAE pre-training,
  CNN-BiLSTM walk-forward training, mean-variance optimization and evaluation.
