# NeuralAlpha pipeline entrypoints.
#
#   make data         download raw data (needs network)
#   make features     build features and targets
#   make tune         tune hyperparameters on the design period
#   make pretrain     MAE self-supervised pre-training
#   make train        walk-forward predictions (supervised baseline)
#   make train-pt     walk-forward fine-tuning from a pre-trained encoder
#   make optimize     top-k ranking and mean-variance backtest
#   make evaluate     metrics, significance and regime tables
#   make reproduce    the full pipeline in order
#
#   make test         run the test suite with a coverage gate
#   make smoke        run the offline end-to-end pipeline test
#   make lint         ruff + black --check + mypy
#   make format       apply ruff --fix and black
#   make clean        remove caches

PY ?= python3
RUN ?= experiments/run
OPT ?= experiments/optimize
PRE ?= experiments/pretrain
REPORTS ?= reports
SEEDS ?= 0,1,2,3,4

.PHONY: help data features tune pretrain train train-pt optimize evaluate reproduce \
        test smoke lint format clean

help:
	@echo "NeuralAlpha targets:"
	@grep -E '^[a-z][a-z-]*:' Makefile | sed 's/^/  /;s/:.*//' | sort -u

data:
	$(PY) scripts/01_fetch_data.py

features:
	$(PY) scripts/02_build_features.py

tune:
	$(PY) scripts/03_train_predict.py --mode tune --seeds $(SEEDS)

pretrain:
	$(PY) scripts/03_train_predict.py --mode pretrain --seeds $(SEEDS)

train:
	$(PY) scripts/03_train_predict.py --out $(RUN) --seeds $(SEEDS)

train-pt:
	$(PY) scripts/03_train_predict.py --mode walk-forward-pretrain \
		--out $(RUN) --seeds $(SEEDS) --pretrained-dir $(PRE)/pretrained

optimize:
	$(PY) scripts/04_optimize.py --predictions $(RUN)/predictions.csv --out $(OPT)

evaluate:
	$(PY) scripts/05_evaluate.py \
		--portfolio $(OPT)/portfolio_returns.csv \
		--baselines $(OPT)/baseline_returns.csv \
		--weights $(OPT)/weights.csv \
		--predictions $(RUN)/predictions.csv \
		--out $(REPORTS)

reproduce: data features train optimize evaluate
	@echo "pipeline finished -> $(REPORTS)"

test:
	$(PY) -m pytest --cov=src/lq45 --cov-fail-under=70 -q

smoke:
	$(PY) -m pytest tests/test_pipeline_smoke.py -q

lint:
	ruff check .
	black --check .
	mypy

format:
	ruff check --fix .
	black .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
