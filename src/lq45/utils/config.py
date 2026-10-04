"""Configuration loading and project file locations."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]

# Data and output locations are overridable so the pipeline can run against a
# sandboxed tree (tests, CI) without touching the working copy. Configs stay
# with the code unless LQ45_CONFIG_DIR says otherwise.
CONFIG_DIR = Path(os.environ.get("LQ45_CONFIG_DIR", ROOT / "configs"))
DATA_DIR = Path(os.environ.get("LQ45_DATA_DIR", ROOT / "data"))
REPORT_DIR = Path(os.environ.get("LQ45_REPORT_DIR", ROOT / "reports"))
EXPERIMENT_DIR = Path(os.environ.get("LQ45_EXPERIMENT_DIR", ROOT / "experiments"))

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


def load_config(name: str) -> dict[str, Any]:
    """Load a single YAML file from ``configs/`` by name, without extension."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"configuration not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return data


def ensure_dirs(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def git_sha() -> str:
    """Short hash of the last commit; `unknown` if not a git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stock_file_name(ticker: str) -> str:
    """File name without the `.JK` suffix."""
    return ticker.replace(".JK", "")
