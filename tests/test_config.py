"""Config helper tests; run directly: python3 tests/test_config.py."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.utils import config


def test_load_config_reads_mapping() -> None:
    loaded = config.load_config("model")
    assert isinstance(loaded, dict)
    assert "lookback_days" in loaded


def test_load_config_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        config.load_config("does_not_exist")


def test_load_config_rejects_non_mapping(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "bad.yaml").write_text("- a\n- b\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError):
        config.load_config("bad")


def test_stock_file_name_strips_suffix() -> None:
    assert config.stock_file_name("BBCA.JK") == "BBCA"
    assert config.stock_file_name("BBCA") == "BBCA"


def test_git_sha_returns_string() -> None:
    sha = config.git_sha()
    assert isinstance(sha, str) and sha


def test_ensure_dirs_creates_nested(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    config.ensure_dirs(target)
    assert target.is_dir()


def main() -> int:
    """Run the fixture-free tests directly; the rest need pytest fixtures."""
    test_load_config_reads_mapping()
    test_stock_file_name_strips_suffix()
    test_git_sha_returns_string()
    print("test_config.py: 3 tests passed (see pytest for the fixture-based ones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
