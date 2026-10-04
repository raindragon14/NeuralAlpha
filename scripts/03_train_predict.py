#!/usr/bin/env python3
"""Stage 3: train CNN-BiLSTM and write walk-forward predictions.

Modes:
- `dry-run`           : print the fold table without training.
- `smoke`             : 2 folds, 1 seed, 3 epochs (pipeline check).
- `calibrate`         : measure seconds per epoch on the smallest and largest fold.
- `tune`              : grid and design-period sensitivity, then freeze the
                        best hyperparameters to `best_hyperparams.yaml`.
- `pretrain`          : Masked Autoencoder pre-training on OHLCV+macro data.
- `walk-forward`      : OOS 2020-2025 predictions for stage 4 (supervised only).
- `walk-forward-pretrain`: pre-train -> fine-tune per fold (two-stage).

Output per run (`experiments/<run_id>/`):
- `run_info.json`      : configuration snapshot, git sha, library versions.
- `fold_metrics.csv`   : loss, best epoch, seconds per epoch per training run.
- `predictions.csv`    : one row per (date, stock, seed); the `role` column
                         marks test or validation.
- `predictions_parts/` : raw parts before deduplication (for audit).
- `checkpoints/`       : state_dict per (fold, seed), optional.
- `log.txt`            : progress with timestamps.
- `pretrained/`        : pre-trained encoder/decoder (pretrain mode).

Prediction deduplication rule: the test windows cover the whole OOS period
without overlap, whereas the validation window of the following fold
overlaps with the test window of the previous fold (step 21 < test 21).
Each (date, stock, seed) is kept from the smallest fold that contains that
date, so there is exactly one row per (date, stock, seed); the recorded
role lives in the `role` column and the stage-5 evaluation filters
`role=test`.

Specification: configs/model.yaml, configs/split.yaml, configs/universe.yaml.
Decision log: docs/keputusan_desain.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from lq45.features.preprocess import RobustPreprocessor
from lq45.models.dataset import (
    FEATURE_COLUMNS,
    PanelData,
    build_pretrain_windows,
    build_windows,
    forward_log_return,
    load_panel,
    pretrain_cutoff_index,
)
from lq45.models.decoder import MAEDecoder
from lq45.models.encoder import CNNBiLSTMEncoder
from lq45.models.pretrain import (
    load_pretrained_encoder,
    pretrain_mae,
)
from lq45.models.training import (
    build_model,
    fine_tune_model,
    predict,
    resolve_device,
    set_seed,
    train_model,
)
from lq45.models.walkforward import Fold, design_split, make_folds
from lq45.utils.config import (
    EXPERIMENT_DIR,
    PROCESSED_DIR,
    ensure_dirs,
    git_sha,
    load_config,
    stock_file_name,
)

DEFAULT_SEEDS = "0,1,2,3,4"


def parse_args() -> argparse.Namespace:
    """Read the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CNN-BiLSTM and write walk-forward predictions."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "walk-forward",
            "walk-forward-pretrain",
            "pretrain",
            "tune",
            "smoke",
            "calibrate",
            "dry-run",
        ],
        default="walk-forward",
        help="operating mode (default: walk-forward)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=f"comma-separated list of seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="number of parallel processes"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch threads per process (0 = automatic)",
    )
    parser.add_argument("--folds", type=int, default=0, help="limit folds (0 = all)")
    parser.add_argument(
        "--epochs", type=int, default=0, help="limit epochs (0 = config)"
    )
    parser.add_argument(
        "--use-tuning",
        type=Path,
        default=None,
        help="tuning result directory used to freeze the hyperparameters",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="do not save state_dict",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip completed (fold, seed) pairs to resume an interrupted " "execution",
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=None,
        help="directory holding encoder_seed{N}.pt (walk-forward-pretrain mode)",
    )
    return parser.parse_args()


def parse_seeds(text: str | None) -> list[int]:
    """Turn a textual seed list into numbers."""
    return [int(x) for x in (text or DEFAULT_SEEDS).split(",") if x.strip()]


def new_run_dir(mode: str) -> Path:
    """Result directory name derived from the UTC timestamp."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return EXPERIMENT_DIR / f"{mode}_{timestamp}"


def log(run_dir: Path, message: str) -> None:
    """Write a single log line to the screen and to `log.txt`."""
    line = f"[{datetime.now(UTC).isoformat()}] {message}"
    print(line, flush=True)
    with (run_dir / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically (temporary file then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    temporary.replace(path)


def load_tuning(path: Path | None) -> dict[str, Any]:
    """Read `best_hyperparams.yaml` from the tuning results; empty if absent."""
    if path is None:
        return {}
    tuning_file = path / "best_hyperparams.yaml"
    if not tuning_file.exists():
        raise FileNotFoundError(f"tuning file not found: {tuning_file}")
    return yaml.safe_load(tuning_file.read_text(encoding="utf-8"))


def model_kwargs(model_cfg: dict[str, Any], tuned: dict[str, Any]) -> dict:
    """Build the CNNBiLSTM constructor parameters from the config.

    Keys present in `tuned` override the config values; `tuned` comes from
    `best_hyperparams.yaml` produced by the design-period tuning.
    """
    return {
        "n_features": len(FEATURE_COLUMNS),
        "filters": list(model_cfg["cnn"]["filters"]),
        "kernel_size": int(model_cfg["cnn"]["kernel_size"]),
        "pooling": int(tuned.get("pooling.size", model_cfg["cnn"]["pooling"]["size"])),
        "units": int(tuned.get("bilstm.units", model_cfg["bilstm"]["units"])),
        "layers": int(model_cfg["bilstm"]["layers"]),
        "batchnorm": bool(model_cfg["cnn"]["batchnorm"]),
        "dropout_cnn": float(model_cfg["dropout"]["cnn"]),
        "dropout_lstm": float(model_cfg["dropout"]["lstm"]),
        "dropout_dense": float(model_cfg["dropout"]["dense"]),
        "bidirectional": bool(model_cfg["bilstm"]["bidirectional"]),
        "activation": model_cfg["cnn"]["activation"],
    }


def train_kwargs(
    model_cfg: dict[str, Any],
    tuned: dict[str, Any],
    epochs: int = 0,
) -> dict:
    """Build the training parameters from the config; `tuned` overrides them."""
    return {
        "batch_size": int(tuned.get("batch_size", model_cfg["batch_size"])),
        "epochs": epochs or int(model_cfg["epochs"]),
        "patience": int(model_cfg["early_stopping"]["patience"]),
        "lr": float(model_cfg["optimizer"]["lr"]),
        "weight_decay": float(
            tuned.get(
                "optimizer.weight_decay",
                model_cfg["optimizer"]["weight_decay"],
            )
        ),
    }


def scale_panel(panel: PanelData, train_span: tuple[int, int]) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    RobustPreprocessor,
    RobustPreprocessor,
]:
    """Winsorize + min-max fitted on the training rows, applied to the whole panel.

    Preprocessing follows Sebastian & Tantia (2024), data preprocessing
    section. The parameters are computed from the date rows of `train_span`
    pooled across all stocks, consistent with the shared model; the
    transform works element-wise, so applying it before window construction
    gives the same result as applying it to each window. Fitting on the
    training data only prevents information leakage.
    """
    a, b = train_span
    train_features = np.concatenate(
        [panel.features[t][a:b] for t in panel.tickers], axis=0
    )
    train_target = np.concatenate([panel.target[t][a:b] for t in panel.tickers], axis=0)
    feature_prep = RobustPreprocessor().fit(
        pd.DataFrame(train_features, columns=FEATURE_COLUMNS)
    )
    target_prep = RobustPreprocessor().fit(pd.DataFrame({"target": train_target}))

    features = {
        t: feature_prep.transform(
            pd.DataFrame(panel.features[t], columns=FEATURE_COLUMNS)
        ).to_numpy(dtype=np.float32)
        for t in panel.tickers
    }
    target = {
        t: target_prep.transform(pd.DataFrame({"target": panel.target[t]}))[
            "target"
        ].to_numpy(dtype=np.float32)
        for t in panel.tickers
    }
    return features, target, feature_prep, target_prep


def window_pool(
    panel: PanelData,
    features: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    span: tuple[int, int],
    banned: list[tuple[int, int]],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the windows of all stocks for a given date span."""
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for ticker in panel.tickers:
        x, y, _ = build_windows(
            features[ticker],
            target[ticker],
            lookback,
            span[0],
            span[1],
            banned,
        )
        if len(y):
            x_list.append(x)
            y_list.append(y)
    if not x_list:
        empty = np.empty((0, len(FEATURE_COLUMNS), lookback), dtype=np.float32)
        return empty, np.empty(0, dtype=np.float32)
    return np.concatenate(x_list), np.concatenate(y_list)


def run_fit(
    panel: PanelData,
    fold: Fold,
    lookback: int,
    model_kw: dict[str, Any],
    train_kw: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
    save_checkpoint: bool,
    fine_tune: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one (fold, seed) then write the prediction part and metrics.

    With `fine_tune` unset the model trains from scratch. Otherwise the
    pre-trained `encoder_seed{N}.pt` is loaded from
    `fine_tune["pretrained_dir"]` and only fine-tuned. Both paths share the
    prediction-row builder and the metrics below.
    """
    start = time.time()
    features, target, _, target_prep = scale_panel(panel, fold.train)
    x_train, y_train = window_pool(
        panel, features, target, fold.train, fold.banned, lookback
    )
    x_val, y_val = window_pool(panel, features, target, fold.validation, [], lookback)
    x_test, y_test = window_pool(panel, features, target, fold.test, [], lookback)

    model = build_model(model_kw, seed)
    if fine_tune is None:
        result = train_model(
            model,
            (x_train, y_train),
            (x_val, y_val),
            train_kw["batch_size"],
            train_kw["epochs"],
            train_kw["patience"],
            train_kw["lr"],
            train_kw["weight_decay"],
            seed,
            device,
        )
    else:
        pretrained_dir = fine_tune["pretrained_dir"]
        pretrained_path = pretrained_dir / f"encoder_seed{seed}.pt"
        if not pretrained_path.exists():
            available = sorted(p.name for p in pretrained_dir.glob("encoder_seed*.pt"))
            raise FileNotFoundError(
                f"encoder_seed{seed}.pt is missing in {pretrained_dir}. "
                f"Available: {available}"
            )
        load_pretrained_encoder(
            model.encoder, torch.load(pretrained_path, map_location=device)
        )
        result = fine_tune_model(
            model,
            (x_train, y_train),
            (x_val, y_val),
            train_kw["batch_size"],
            train_kw["epochs"],
            train_kw["patience"],
            fine_tune["head_lr"],
            fine_tune["encoder_lr"],
            fine_tune["weight_decay"],
            seed,
            device,
            freeze_encoder=fine_tune["freeze_encoder"],
        )

    pred_test = predict(model, x_test, train_kw["batch_size"], device)
    test_loss = float(np.mean((pred_test - y_test) ** 2))

    rows: list[dict[str, Any]] = []
    for span, role in (
        (fold.validation, "validation"),
        (fold.test, "test"),
    ):
        for ticker in panel.tickers:
            x, y, idx = build_windows(
                features[ticker],
                target[ticker],
                lookback,
                span[0],
                span[1],
                [],
            )
            if len(y) == 0:
                continue
            pred = predict(model, x, train_kw["batch_size"], device)
            pred_raw = target_prep.inverse_transform(pd.DataFrame({"target": pred}))[
                "target"
            ].to_numpy()
            y_raw = target_prep.inverse_transform(pd.DataFrame({"target": y}))[
                "target"
            ].to_numpy()
            for j, position in enumerate(idx):
                rows.append(
                    {
                        "fold": fold.id,
                        "seed": seed,
                        "date": panel.dates[position].date().isoformat(),
                        "ticker": ticker,
                        "role": role,
                        "pred_scaled": float(pred[j]),
                        "true_scaled": float(y[j]),
                        "pred_raw": float(pred_raw[j]),
                        "true_raw": float(y_raw[j]),
                    }
                )

    part = pd.DataFrame(
        rows,
        columns=[
            "fold",
            "seed",
            "date",
            "ticker",
            "role",
            "pred_scaled",
            "true_scaled",
            "pred_raw",
            "true_raw",
        ],
    )
    part.to_csv(
        run_dir / "predictions_parts" / f"{fold.id:03d}_{seed}.csv",
        index=False,
    )
    if save_checkpoint:
        torch.save(
            result.state_dict,
            run_dir / "checkpoints" / f"{fold.id:03d}_{seed}.pt",
        )

    metrics = {
        "fold": fold.id,
        "seed": seed,
        "train_loss": result.history[-1]["train_loss"],
        "val_loss": result.best_val_loss,
        "test_loss": test_loss,
        "best_epoch": result.best_epoch,
        "stopped_early": result.stopped_early,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "wall_sec": round(time.time() - start, 3),
        "sec_per_epoch": round((time.time() - start) / max(len(result.history), 1), 3),
    }
    write_json(run_dir / "metrics" / f"{fold.id:03d}_{seed}.json", metrics)
    return metrics


def train_design(
    panel: PanelData,
    design: Any,
    lookback: int,
    model_kw: dict[str, Any],
    train_kw: dict[str, Any],
    seed: int,
    device: torch.device,
) -> float:
    """Train on the design-period split; return the validation loss.

    Purge is already attached to `design.train`; the target was built
    beforehand by the caller according to the horizon under test.
    """
    features, target, _, _ = scale_panel(panel, design.train)
    x_train, y_train = window_pool(
        panel, features, target, design.train, design.banned, lookback
    )
    x_val, y_val = window_pool(panel, features, target, design.validation, [], lookback)
    model = build_model(model_kw, seed)
    result = train_model(
        model,
        (x_train, y_train),
        (x_val, y_val),
        train_kw["batch_size"],
        train_kw["epochs"],
        train_kw["patience"],
        train_kw["lr"],
        train_kw["weight_decay"],
        seed,
        device,
    )
    return result.best_val_loss


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a nested grid into a list of combinations."""
    combinations: list[dict[str, Any]] = [{}]
    for key, value in grid.items():
        combinations = [{**c, key: v} for c in combinations for v in value]
    return combinations


def is_fold_done(run_dir: Path, fold_id: int, seed: int) -> bool:
    """True if the prediction part and metrics for this pair already exist."""
    part = run_dir / "predictions_parts" / f"{fold_id:03d}_{seed}.csv"
    metrics = run_dir / "metrics" / f"{fold_id:03d}_{seed}.json"
    return part.exists() and metrics.exists()


def load_panel_data(tickers: list[str], horizon: int) -> PanelData:
    """Load the panel; stocks without a processed file are skipped with a warning.

    The official universe holds 45 constituents, but two of them (SRIL,
    WSKT) are unavailable on Yahoo Finance; the effective list is recorded
    in `run_info.json` through `panel.tickers`.
    """
    directory = PROCESSED_DIR / "features"
    available = [
        t for t in tickers if (directory / f"{stock_file_name(t)}.csv").exists()
    ]
    missing = [t for t in tickers if t not in available]
    if missing:
        print(
            f"warning: {len(missing)} stocks without a processed file "
            f"skipped: {missing}"
        )
    if not available:
        raise RuntimeError("no stock with a processed file")
    return load_panel(directory, available, horizon)


def merge_predictions(run_dir: Path) -> pd.DataFrame:
    """Merge the prediction parts and metrics into the final outputs.

    Deduplication: each (date, stock, seed) is kept as a single row from the
    smallest fold that contains that date. The first OOS date appears in the
    validation window of fold 0, the rest appear in the test window of the
    fold that covers them; the validation window of the following fold that
    overlaps the previous test is removed by itself.
    """
    metric_files = sorted((run_dir / "metrics").glob("*.json"))
    if not metric_files:
        raise RuntimeError(f"no metrics in {run_dir}")
    metrics = pd.DataFrame(
        [json.loads(p.read_text(encoding="utf-8")) for p in metric_files]
    )
    metrics.sort_values(["fold", "seed"]).to_csv(
        run_dir / "fold_metrics.csv", index=False
    )

    part_files = sorted((run_dir / "predictions_parts").glob("*.csv"))
    part = pd.concat([pd.read_csv(p) for p in part_files], ignore_index=True)
    part = part.sort_values("fold")
    unique_rows = part.drop_duplicates(subset=["date", "ticker", "seed"], keep="first")
    unique_rows = unique_rows.sort_values(["date", "ticker", "seed"]).reset_index(
        drop=True
    )
    unique_rows.to_csv(run_dir / "predictions.csv", index=False)
    return unique_rows


def smoke_summary(run_dir: Path, predictions: pd.DataFrame) -> None:
    """Concise initial check for smoke mode.

    The correlation between predictions and actual values in the stock
    market tends to be small; it is printed so anomalies (for example data
    leakage) become immediately visible.
    """
    correlation = predictions[["pred_raw", "true_raw"]].corr().iloc[0, 1]
    mse_raw = float(np.mean((predictions["pred_raw"] - predictions["true_raw"]) ** 2))
    print(
        f"initial check: correlation between predictions and actual = {correlation:.4f} "
        f"(a small value is expected), raw-space MSE = {mse_raw:.6f}"
    )
    log(
        run_dir,
        f"initial check: correlation {correlation:.4f}, raw-space MSE {mse_raw:.6f}",
    )


def write_run_info(
    run_dir: Path,
    mode: str,
    seeds: list[int],
    jobs: int,
    threads: int,
    lookback: int,
    horizon: int,
    n_folds: int,
    device: str,
    use_tuning: Path | None,
) -> None:
    """Write the execution record for auditing."""
    info = {
        "mode": mode,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "seeds": seeds,
        "jobs": jobs,
        "threads": threads,
        "lookback_days": lookback,
        "horizon_days": horizon,
        "n_folds": n_folds,
        "device": device,
        "use_tuning": str(use_tuning) if use_tuning else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "configs": {
            name: load_config(name)
            for name in ("data", "split", "model", "portfolio", "experiment")
        },
    }
    write_json(run_dir / "run_info.json", info)


def run_dry_run(args: argparse.Namespace) -> int:
    """Print the fold table without training."""
    model_cfg = load_config("model")
    split_cfg = load_config("split")
    tickers = list(load_config("universe")["tickers"])
    tuned = load_tuning(args.use_tuning)
    lookback = int(tuned.get("lookback_days", model_cfg["lookback_days"]))
    horizon = int(tuned.get("horizon_days", model_cfg["horizon_days"]))
    panel = load_panel_data(tickers, horizon)
    folds = make_folds(
        len(panel.dates),
        split_cfg,
        purge_days=horizon,
        embargo_days=lookback - 1,
    )
    if args.folds:
        folds = folds[: args.folds]

    print(
        f"calendar: {panel.dates[0].date()} to {panel.dates[-1].date()}"
        f" ({len(panel.dates)} days), w={lookback}, tau={horizon}"
    )
    header = (
        f"{'id':>3} {'train up to':>12} {'validation':>23} {'test':>23} "
        f"{'n_train':>9} {'n_val':>7} {'n_test':>7} {'banned':>9}"
    )
    print(header)
    for fold in folds:
        n_train = sum(
            len(
                build_windows(
                    panel.features[t],
                    panel.target[t],
                    lookback,
                    fold.train[0],
                    fold.train[1],
                    fold.banned,
                )[1]
            )
            for t in panel.tickers
        )
        n_val = sum(
            len(
                build_windows(
                    panel.features[t],
                    panel.target[t],
                    lookback,
                    fold.validation[0],
                    fold.validation[1],
                    [],
                )[1]
            )
            for t in panel.tickers
        )
        n_test = sum(
            len(
                build_windows(
                    panel.features[t],
                    panel.target[t],
                    lookback,
                    fold.test[0],
                    fold.test[1],
                    [],
                )[1]
            )
            for t in panel.tickers
        )
        span = lambda ab: (
            f"{panel.dates[ab[0]].date()} - {panel.dates[ab[1] - 1].date()}"
        )
        print(
            f"{fold.id:>3} {panel.dates[fold.train[1] - 1].date():>12} "
            f"{span(fold.validation):>23} {span(fold.test):>23} "
            f"{n_train:>9} {n_val:>7} {n_test:>7} {len(fold.banned):>9}"
        )
    print(f"number of folds: {len(folds)}")
    return 0


def run_tune(args: argparse.Namespace) -> int:
    """Tune hyperparameters on the design period (grid + sensitivity)."""
    model_cfg = load_config("model")
    tickers = list(load_config("universe")["tickers"])
    seeds = parse_seeds(args.seeds)
    device = resolve_device(model_cfg["device"])
    torch.set_num_threads(args.threads or torch.get_num_threads())

    run_dir = args.out or new_run_dir("tune")
    ensure_dirs(run_dir)
    log(run_dir, f"tuning started: {len(seeds)} seeds, device {device}")

    panel = load_panel_data(tickers, 5)
    tune_cfg = model_cfg["tuning"]
    design = design_split(
        panel.dates, tune_cfg["train"], tune_cfg["validate"], horizon=5
    )
    log(
        run_dir,
        f"design period: train {panel.dates[design.train[0]].date()} to "
        f"{panel.dates[design.train[1] - 1].date()}; validation "
        f"{panel.dates[design.validation[0]].date()} to "
        f"{panel.dates[design.validation[1] - 1].date()}",
    )

    grid = expand_grid(tune_cfg["grid"])
    rows: list[dict[str, Any]] = []
    for i, tuned in enumerate(grid, 1):
        model_kw = model_kwargs(model_cfg, tuned)
        train_kw = train_kwargs(model_cfg, tuned)
        scores = [
            train_design(panel, design, 60, model_kw, train_kw, seed, device)
            for seed in seeds
        ]
        rows.append(
            {
                **tuned,
                "val_loss_mean": float(np.mean(scores)),
                "val_loss_std": float(np.std(scores)),
            }
        )
        log(
            run_dir,
            f"grid {i}/{len(grid)}: {tuned} -> mean {np.mean(scores):.6f}",
        )
    grid_df = pd.DataFrame(rows).sort_values("val_loss_mean")
    grid_df.to_csv(run_dir / "grid_results.csv", index=False)
    best_hp = {
        key: value for key, value in grid_df.iloc[0].items() if key in tune_cfg["grid"]
    }
    log(run_dir, f"best grid: {best_hp}")

    model_kw = model_kwargs(model_cfg, best_hp)
    train_kw = train_kwargs(model_cfg, best_hp)
    sensitivity_rows: list[dict[str, Any]] = []
    sens_cfg = model_cfg.get("sensitivity", {})
    for w in sens_cfg.get("lookback_days", [60]):
        for tau in sens_cfg.get("horizon_days", [5]):
            target = {t: forward_log_return(panel.close[t], tau) for t in panel.tickers}
            panel_tau = dataclasses.replace(panel, target=target)
            design_tau = design_split(
                panel.dates, tune_cfg["train"], tune_cfg["validate"], tau
            )
            scores = [
                train_design(panel_tau, design_tau, w, model_kw, train_kw, seed, device)
                for seed in seeds
            ]
            sensitivity_rows.append(
                {
                    "lookback_days": w,
                    "horizon_days": tau,
                    "val_loss_mean": float(np.mean(scores)),
                    "val_loss_std": float(np.std(scores)),
                }
            )
            log(
                run_dir,
                f"sensitivity w={w} tau={tau} -> " f"mean {np.mean(scores):.6f}",
            )
    sens_df = pd.DataFrame(sensitivity_rows).sort_values("val_loss_mean")
    sens_df.to_csv(run_dir / "sensitivity_results.csv", index=False)

    best_grid = dict(grid_df.iloc[0])
    best_window = dict(sens_df.iloc[0])
    best = {
        "bilstm.units": int(best_grid["bilstm.units"]),
        "batch_size": int(best_grid["batch_size"]),
        "optimizer.weight_decay": float(best_grid["optimizer.weight_decay"]),
        "pooling.size": int(best_grid["pooling.size"]),
        "lookback_days": int(best_window["lookback_days"]),
        "horizon_days": int(best_window["horizon_days"]),
        "grid_val_loss_mean": float(best_grid["val_loss_mean"]),
        "sensitivity_val_loss_mean": float(best_window["val_loss_mean"]),
        "seeds": seeds,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
    }
    with (run_dir / "best_hyperparams.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best, handle, sort_keys=False, allow_unicode=True)
    log(run_dir, f"best hyperparameters frozen: {best}")
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    """Measure the training time on the smallest and largest fold."""
    model_cfg = load_config("model")
    split_cfg = load_config("split")
    tickers = list(load_config("universe")["tickers"])
    tuned = load_tuning(args.use_tuning)
    lookback = int(tuned.get("lookback_days", model_cfg["lookback_days"]))
    horizon = int(tuned.get("horizon_days", model_cfg["horizon_days"]))
    model_kw = model_kwargs(model_cfg, tuned)
    train_kw = train_kwargs(model_cfg, tuned, args.epochs)
    device = resolve_device(model_cfg["device"])
    torch.set_num_threads(args.threads or torch.get_num_threads())

    panel = load_panel_data(tickers, horizon)
    folds = make_folds(
        len(panel.dates),
        split_cfg,
        purge_days=horizon,
        embargo_days=lookback - 1,
    )
    run_dir = args.out or new_run_dir("calibrate")
    ensure_dirs(
        run_dir / "predictions_parts",
        run_dir / "checkpoints",
        run_dir / "metrics",
    )
    log(run_dir, f"calibration started: {len(folds)} folds available")

    measurements: list[dict[str, Any]] = []
    for fold in (folds[0], folds[-1]):
        metrics = run_fit(
            panel, fold, lookback, model_kw, train_kw, 0, device, run_dir, False
        )
        measurements.append(metrics)
        log(
            run_dir,
            f"fold {fold.id}: time {metrics['wall_sec']} seconds, "
            f"{metrics['sec_per_epoch']} sec/epoch, "
            f"{metrics['best_epoch']} epochs",
        )
    pd.DataFrame(measurements).to_csv(run_dir / "calibrate.csv", index=False)

    n_seeds = len(parse_seeds(args.seeds))
    n_fits = len(folds) * n_seeds
    fast_total = measurements[0]["wall_sec"] * n_fits
    slow_total = measurements[1]["wall_sec"] * n_fits
    log(
        run_dir,
        f"extrapolating {n_fits} training runs: "
        f"{fast_total / 3600:.1f} hours (first-fold rate) to "
        f"{slow_total / 3600:.1f} hours (last-fold rate)",
    )
    print(
        f"estimated total time: {fast_total / 3600:.1f} - "
        f"{slow_total / 3600:.1f} hours for {n_fits} training runs "
        f"({len(folds)} folds x {n_seeds} seeds)"
    )
    return 0


def run_pretrain(args: argparse.Namespace) -> int:
    """Masked Autoencoder pre-training on the processed feature panel.

    Self-supervised pre-training without return labels on the same 8
    FEATURE_COLUMNS the supervised model consumes, so the pre-trained
    encoder transfers into every walk-forward fold unchanged.
    """
    model_cfg = load_config("model")
    pretrain_cfg = model_cfg.get("pretrain", {})

    tickers = list(load_config("universe")["tickers"])
    seeds = parse_seeds(args.seeds)
    device = resolve_device(model_cfg["device"])
    torch.set_num_threads(args.threads or torch.get_num_threads())

    run_dir = args.out or new_run_dir("pretrain")
    ensure_dirs(
        run_dir / "pretrained",
        run_dir / "metrics",
    )
    log(run_dir, f"MAE pre-training started: {len(seeds)} seeds, device {device}")

    if not pretrain_cfg.get("enabled", False):
        log(run_dir, "pre-training disabled in config (pretrain.enabled=false)")
        return 1

    # Load the processed feature panel (FEATURE_COLUMNS) so the pre-training
    # input matches the supervised model exactly and the encoder transfers.
    panel = load_panel_data(tickers, int(model_cfg["horizon_days"]))
    log(run_dir, f"loaded {len(panel.tickers)} stocks for pre-training")

    lookback = int(model_cfg["lookback_days"])
    n_features = len(FEATURE_COLUMNS)
    cutoff_index = pretrain_cutoff_index(panel.dates, pretrain_cfg.get("data_end"))
    log(
        run_dir,
        f"pre-training data cutoff: index {cutoff_index} "
        f"({pretrain_cfg.get('data_end') or 'none'})",
    )

    # First pass: build pre-training windows with a window-level train/val
    # split, so the short pre-OOS series still yields a non-empty val set.
    stock_windows = []
    for ticker in panel.tickers:
        features = panel.features[ticker][:cutoff_index]
        windows, _ = build_pretrain_windows(features, lookback, 0, len(features))
        n_val = int(len(windows) * pretrain_cfg.get("val_ratio", 0.1))
        stock_windows.append((windows, len(windows) - n_val))

    # Fit RobustPreprocessor on the training windows (winsorize + min-max)
    train_flats = [
        w[:n].transpose(0, 2, 1).reshape(-1, n_features)
        for w, n in stock_windows
        if n > 0
    ]
    if train_flats:
        combined_train = np.concatenate(train_flats, axis=0)
        prep = RobustPreprocessor().fit(
            pd.DataFrame(combined_train, columns=FEATURE_COLUMNS)
        )
        log(
            run_dir,
            f"RobustPreprocessor fitted on {len(combined_train)} training samples",
        )
    else:
        log(
            run_dir,
            "WARNING: no training data for preprocessor, skipping normalization",
        )
        prep = None

    # Second pass: transform the windows and split them into train/val parts
    all_train_x = []
    all_val_x = []
    for windows, n_train in stock_windows:
        flat = windows.transpose(0, 2, 1).reshape(-1, n_features)
        if prep is not None:
            flat = prep.transform(
                pd.DataFrame(flat, columns=FEATURE_COLUMNS)
            ).to_numpy(dtype=np.float32)
        transformed = (
            flat.reshape(len(windows), lookback, n_features)
            .transpose(0, 2, 1)
            .astype(np.float32)
        )
        if n_train > 0:
            all_train_x.append(transformed[:n_train])
        if len(transformed) > n_train:
            all_val_x.append(transformed[n_train:])

    train_x = (
        np.concatenate(all_train_x, axis=0)
        if all_train_x
        else np.empty((0, n_features, lookback), dtype=np.float32)
    )
    val_x = (
        np.concatenate(all_val_x, axis=0)
        if all_val_x
        else np.empty((0, n_features, lookback), dtype=np.float32)
    )

    log(run_dir, f"pre-train samples: train {len(train_x)}, val {len(val_x)}")

    if len(train_x) == 0:
        log(run_dir, "no pre-train data")
        return 1

    # Run pre-training for each seed
    for seed in seeds:
        log(run_dir, f"pre-train seed {seed}...")
        set_seed(seed)

        encoder = CNNBiLSTMEncoder(
            n_features=n_features,
            filters=list(model_cfg["cnn"]["filters"]),
            kernel_size=int(model_cfg["cnn"]["kernel_size"]),
            pooling=int(model_cfg["cnn"]["pooling"]["size"]),
            units=int(model_cfg["bilstm"]["units"]),
            layers=int(model_cfg["bilstm"]["layers"]),
            batchnorm=bool(model_cfg["cnn"]["batchnorm"]),
            dropout_cnn=float(model_cfg["dropout"]["cnn"]),
            dropout_lstm=float(model_cfg["dropout"]["lstm"]),
            bidirectional=bool(model_cfg["bilstm"]["bidirectional"]),
            activation=model_cfg["cnn"]["activation"],
        )
        decoder = MAEDecoder(
            latent_dim=encoder.output_dim,
            n_channels=n_features,
            lookback=lookback,
            pooling=int(model_cfg["cnn"]["pooling"]["size"]),
            hidden=pretrain_cfg.get("decoder", {}).get("hidden", 128),
            layers=pretrain_cfg.get("decoder", {}).get("layers", 2),
        )

        result = pretrain_mae(
            encoder=encoder,
            decoder=decoder,
            train_x=train_x,
            val_x=val_x,
            mask_ratio=pretrain_cfg.get("mask_ratio", 0.3),
            epochs=args.epochs or pretrain_cfg.get("epochs", 50),
            lr=pretrain_cfg.get("lr", 1e-3),
            weight_decay=pretrain_cfg.get("weight_decay", 0.0),
            batch_size=pretrain_cfg.get("batch_size", 64),
            patience=pretrain_cfg.get("patience", 10),
            device=device,
            seed=seed,
        )

        # Save pretrained encoder/decoder
        torch.save(
            result.encoder_state, run_dir / "pretrained" / f"encoder_seed{seed}.pt"
        )
        torch.save(
            result.decoder_state, run_dir / "pretrained" / f"decoder_seed{seed}.pt"
        )

        # Save metrics
        metrics = {
            "seed": seed,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "wall_sec": result.wall_sec,
        }
        write_json(run_dir / "metrics" / f"pretrain_seed{seed}.json", metrics)
        log(
            run_dir,
            f"seed {seed}: val_loss {result.best_val_loss:.6f}, "
            f"epoch {result.best_epoch}",
        )

    # Save combined best (seed 0 as reference)
    write_json(
        run_dir / "metrics" / "pretrain_summary.json",
        {
            "seeds": seeds,
            "lookback": lookback,
            "pretrain_channels": list(FEATURE_COLUMNS),
            "config": pretrain_cfg,
        },
    )
    log(run_dir, f"pre-training done -> {run_dir / 'pretrained'}")
    return 0


@dataclass
class _WorkerContext:
    """Worker process context; filled in by the initializer function."""

    panel: PanelData | None = None
    folds: list[Fold] | None = None
    lookback: int | None = None
    model_kw: dict[str, Any] | None = None
    train_kw: dict[str, Any] | None = None
    device: torch.device | None = None
    run_dir: Path | None = None
    save_checkpoint: bool | None = None
    fine_tune: dict[str, Any] | None = None


_worker: _WorkerContext | None = None


def _init_worker(
    panel: PanelData,
    folds: list[Fold],
    lookback: int,
    model_kw: dict[str, Any],
    train_kw: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    save_checkpoint: bool,
    fine_tune: dict[str, Any] | None,
    thread: int,
) -> None:
    """Initialize the parallel worker process."""
    global _worker
    torch.set_num_threads(thread)
    _worker = _WorkerContext()
    _worker.panel = panel
    _worker.folds = folds
    _worker.lookback = lookback
    _worker.model_kw = model_kw
    _worker.train_kw = train_kw
    _worker.device = device
    _worker.run_dir = run_dir
    _worker.save_checkpoint = save_checkpoint
    _worker.fine_tune = fine_tune


def _fit_task(pairs: tuple[int, int]) -> tuple[int, int]:
    """Run one (fold, seed) in the worker process."""
    fold_id, seed = pairs
    fold = _worker.folds[fold_id]
    run_fit(
        _worker.panel,
        fold,
        _worker.lookback,
        _worker.model_kw,
        _worker.train_kw,
        seed,
        _worker.device,
        _worker.run_dir,
        _worker.save_checkpoint,
        fine_tune=_worker.fine_tune,
    )
    return fold_id, seed


def run_walkforward(
    args: argparse.Namespace, pretrained: bool = False
) -> int:
    """Run the full walk-forward (or smoke) and write the predictions.

    With `pretrained=True`, each fold fine-tunes the pre-trained
    `encoder_seed{N}.pt` (from `--pretrained-dir`, default
    `run_dir/pretrained`) instead of training from scratch; pre-training is
    NOT run automatically.
    """
    model_cfg = load_config("model")
    pretrain_cfg = model_cfg.get("pretrain", {})
    split_cfg = load_config("split")
    tickers = list(load_config("universe")["tickers"])

    if args.mode == "smoke":
        seeds = parse_seeds(args.seeds)[:1]
        fold_limit = args.folds or 2
        epoch_limit = args.epochs or 3
    else:
        seeds = parse_seeds(args.seeds)
        fold_limit = args.folds
        epoch_limit = args.epochs

    tuned = load_tuning(args.use_tuning)
    lookback = int(tuned.get("lookback_days", model_cfg["lookback_days"]))
    horizon = int(tuned.get("horizon_days", model_cfg["horizon_days"]))
    model_kw = model_kwargs(model_cfg, tuned)
    train_kw = train_kwargs(model_cfg, tuned, epoch_limit)

    device = resolve_device(model_cfg["device"])
    threads = args.threads or torch.get_num_threads()
    if args.jobs > 1 and not args.threads:
        threads = max(1, min(4, (os.cpu_count() or 1) // args.jobs))
    torch.set_num_threads(threads)

    panel = load_panel_data(tickers, horizon)
    folds = make_folds(
        len(panel.dates),
        split_cfg,
        purge_days=horizon,
        embargo_days=lookback - 1,
    )
    if fold_limit:
        folds = folds[:fold_limit]

    run_dir = args.out or new_run_dir(args.mode)
    fine_tune = None
    if pretrained:
        ft_cfg = pretrain_cfg.get("fine_tune", {})
        fine_tune = {
            "pretrained_dir": args.pretrained_dir or (run_dir / "pretrained"),
            "freeze_encoder": ft_cfg.get("freeze_encoder", False),
            "head_lr": ft_cfg.get("head_lr", 1e-4),
            "encoder_lr": ft_cfg.get("encoder_lr", 1e-5),
            "weight_decay": pretrain_cfg.get("weight_decay", 0.0),
        }
    dirs = [run_dir / "predictions_parts", run_dir / "checkpoints", run_dir / "metrics"]
    if pretrained:
        dirs.append(run_dir / "pretrained")
    ensure_dirs(*dirs)
    log(
        run_dir,
        f"starting {args.mode}: {len(folds)} folds, {len(seeds)} seeds, "
        f"jobs {args.jobs}, threads {threads}, device {device}",
    )
    if fine_tune is not None:
        log(run_dir, f"pre-trained encoder read from: {fine_tune['pretrained_dir']}")

    pairs = [(fold.id, seed) for fold in folds for seed in seeds]
    pairs = [
        p for p in pairs if not (args.resume and is_fold_done(run_dir, p[0], p[1]))
    ]
    if not pairs:
        log(run_dir, "all pairs already done (resume mode)")
    elif args.jobs <= 1:
        for fold_id, seed in pairs:
            fold = folds[fold_id]
            metrics = run_fit(
                panel,
                fold,
                lookback,
                model_kw,
                train_kw,
                seed,
                device,
                run_dir,
                not args.no_checkpoints,
                fine_tune=fine_tune,
            )
            log(
                run_dir,
                f"fold {fold.id} seed {seed}: val "
                f"{metrics['val_loss']:.6f}, test {metrics['test_loss']:.6f}, "
                f"time {metrics['wall_sec']} seconds",
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(
                panel,
                folds,
                lookback,
                model_kw,
                train_kw,
                device,
                run_dir,
                not args.no_checkpoints,
                fine_tune,
                max(1, threads),
            ),
        ) as pool:
            for fold_id, seed in pool.map(_fit_task, pairs):
                log(run_dir, f"finished fold {fold_id} seed {seed}")

    predictions = merge_predictions(run_dir)
    write_run_info(
        run_dir,
        args.mode,
        seeds,
        args.jobs,
        threads,
        lookback,
        horizon,
        len(folds),
        str(device),
        args.use_tuning,
    )
    log(
        run_dir,
        f"done: {len(predictions)} prediction rows -> "
        f"{run_dir / 'predictions.csv'}",
    )
    if args.mode == "smoke":
        smoke_summary(run_dir, predictions)
    return 0


def main() -> int:
    """Run the requested mode."""
    args = parse_args()
    if args.mode == "tune":
        return run_tune(args)
    if args.mode == "calibrate":
        return run_calibrate(args)
    if args.mode == "dry-run":
        return run_dry_run(args)
    if args.mode == "pretrain":
        return run_pretrain(args)
    if args.mode == "walk-forward-pretrain":
        return run_walkforward(args, pretrained=True)
    return run_walkforward(args)


if __name__ == "__main__":
    raise SystemExit(main())
