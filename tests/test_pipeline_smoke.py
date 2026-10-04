"""End-to-end offline smoke test for pipeline stages 02-05.

Runs the real CLI scripts as subprocesses against synthetic data in a temporary
tree, so the whole pipeline is verifiable without Yahoo Finance or Bank
Indonesia access. The market data is synthetic; this validates the plumbing,
not the research results.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from synthetic_data import TICKERS, write_raw

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _run(script: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return result


def test_pipeline_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    experiment_dir = tmp_path / "experiments"
    write_raw(data_dir)

    # Sandbox the configs too, with a universe limited to the synthetic tickers.
    config_dir = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", config_dir)
    (config_dir / "universe.yaml").write_text(
        yaml.safe_dump({"tickers": list(TICKERS)}), encoding="utf-8"
    )

    env = {
        **os.environ,
        "LQ45_DATA_DIR": str(data_dir),
        "LQ45_REPORT_DIR": str(report_dir),
        "LQ45_EXPERIMENT_DIR": str(experiment_dir),
        "LQ45_CONFIG_DIR": str(config_dir),
    }

    _run("02_build_features.py", env=env)
    processed = data_dir / "processed" / "features"
    assert len(list(processed.glob("*.csv"))) == 6

    run_dir = experiment_dir / "smoke"
    _run(
        "03_train_predict.py",
        "--mode",
        "smoke",
        "--seeds",
        "0",
        "--jobs",
        "1",
        "--folds",
        "1",
        "--epochs",
        "1",
        "--out",
        str(run_dir),
        env=env,
    )
    predictions = run_dir / "predictions.csv"
    assert predictions.exists()

    # Two-stage path: MAE pre-training, then fine-tuning per fold.
    pretrain_dir = experiment_dir / "pretrain"
    _run(
        "03_train_predict.py",
        "--mode",
        "pretrain",
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--out",
        str(pretrain_dir),
        env=env,
    )
    assert (pretrain_dir / "pretrained" / "encoder_seed0.pt").exists()

    finetuned_dir = experiment_dir / "smoke-pretrain"
    _run(
        "03_train_predict.py",
        "--mode",
        "walk-forward-pretrain",
        "--seeds",
        "0",
        "--jobs",
        "1",
        "--folds",
        "1",
        "--epochs",
        "1",
        "--pretrained-dir",
        str(pretrain_dir / "pretrained"),
        "--out",
        str(finetuned_dir),
        env=env,
    )
    assert (finetuned_dir / "predictions.csv").exists()

    optimize_dir = experiment_dir / "optimize"
    _run(
        "04_optimize.py",
        "--predictions",
        str(predictions),
        "--grid",
        "main",
        "--seed-mode",
        "ensemble",
        "--out",
        str(optimize_dir),
        env=env,
    )
    assert (optimize_dir / "weights.csv").exists()
    assert (optimize_dir / "portfolio_returns.csv").exists()
    assert (optimize_dir / "baseline_returns.csv").exists()

    _run(
        "05_evaluate.py",
        "--portfolio",
        str(optimize_dir / "portfolio_returns.csv"),
        "--baselines",
        str(optimize_dir / "baseline_returns.csv"),
        "--weights",
        str(optimize_dir / "weights.csv"),
        "--predictions",
        str(predictions),
        "--out",
        str(report_dir),
        env=env,
    )
    for name in (
        "metrics_table.csv",
        "regime_table.csv",
        "ranker_table.csv",
        "dsr_pbo.json",
    ):
        assert (report_dir / name).exists(), f"missing {name}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
