# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pre-commit install
```

A fully pinned environment (Python 3.13, Linux) is in `requirements.lock`.

## Before you push

Activate the virtualenv first: the hooks call `ruff`, `black` and `mypy`
from the active environment (`language: system`).

```bash
source .venv/bin/activate
pre-commit run --all-files   # ruff, black, mypy, whitespace
pytest --cov=src/lq45 --cov-fail-under=70
```

CI runs the same checks on Python 3.12 and 3.13.

## Layout

| Path | Contents |
|---|---|
| `configs/` | YAML config; every key is read by code (no dead keys) |
| `src/lq45/` | package: `data`, `features`, `models`, `portfolio`, `evaluation`, `utils` |
| `scripts/` | numbered pipeline stages `01`–`05` |
| `tests/` | unit tests; `tests/fixtures/` feeds the offline smoke test |
| `docs/` | `keputusan_desain.md` (decision log), `references.bib` |
| `reports/` | generated output (gitignored except `.gitkeep` and `*.md`) |

## Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): imperative subject`, lowercase (e.g.
  `fix(evaluation): correct the DSR trial variance`).
- Every research decision goes in `docs/keputusan_desain.md` with its
  literature basis and the code location that implements it.
- New citations must be verified against the DOI or arXiv record before they
  are added to `docs/references.bib`; do not write bibliographic metadata from
  memory.
- Tests are deterministic: seed everything, no network access.
