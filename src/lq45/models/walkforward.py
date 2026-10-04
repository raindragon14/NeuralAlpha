"""Nested walk-forward fold generator with purge and embargo.

Operational definitions follow Lopez de Prado (2018) Chapter 7:
- Purge: a training sample at date `t` uses label `t + tau`; the label
  must not fall inside the evaluation period, so the training window is
  truncated to `[0, validation_start - tau)`.
- Embargo: each past test period produces a ban of `lookback - 1` days
  after the test ends; if the training window expands to absorb that
  region, samples inside it are dropped.

All boundaries use exchange-calendar positions (indices), not dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252


@dataclass
class Fold:
    """A single walk-forward fold; boundaries are calendar indices."""

    id: int
    train: tuple[int, int]  # [start, end); purge already applied
    validation: tuple[int, int]
    test: tuple[int, int]
    banned: list[tuple[int, int]]  # embargo from past tests


@dataclass
class DesignSplit:
    """Design-period split: train then validation (no test)."""

    train: tuple[int, int]
    validation: tuple[int, int]
    banned: list[tuple[int, int]] = field(default_factory=list)


def make_folds(
    n_days: int,
    split_cfg: dict[str, Any],
    purge_days: int | None = None,
    embargo_days: int | None = None,
) -> list[Fold]:
    """Build all folds from a calendar `n_days` long.

    If `purge_days` and `embargo_days` are not given, they are taken from
    `split_cfg` (defaults 5 and 59). Folds are generated as long as the
    test period still fits inside the calendar.
    """
    initial_train = int(split_cfg["initial_train_years"]) * DAYS_PER_YEAR
    val = int(split_cfg["validation_days"])
    test = int(split_cfg["test_days"])
    step = int(split_cfg["step_days"])
    if purge_days is None:
        purge_days = int(split_cfg["purge"]["horizon_days"])
    if embargo_days is None:
        embargo_days = int(split_cfg["embargo_days"])

    folds: list[Fold] = []
    previous_test_ends: list[int] = []
    k = 0
    while True:
        val_start = initial_train + k * step
        test_start = val_start + val
        test_end = test_start + test
        if test_end > n_days:
            break
        train = (0, max(0, val_start - purge_days))
        banned = [(end, min(end + embargo_days, n_days)) for end in previous_test_ends]
        folds.append(
            Fold(
                id=k,
                train=train,
                validation=(val_start, test_start),
                test=(test_start, test_end),
                banned=banned,
            )
        )
        previous_test_ends.append(test_end)
        k += 1
    return folds


def design_split(
    dates: pd.DatetimeIndex,
    train_range: str,
    validate_range: str,
    horizon: int,
) -> DesignSplit:
    """Design-period split from `YYYY-MM-DD/YYYY-MM-DD` ranges.

    The purge truncates the last `horizon` training days so the label does
    not fall inside the validation period.
    """
    train_start, train_end = (pd.Timestamp(bound) for bound in train_range.split("/"))
    val_start, val_end = (pd.Timestamp(bound) for bound in validate_range.split("/"))

    train_positions = np.flatnonzero((dates >= train_start) & (dates <= train_end))
    validation_positions = np.flatnonzero((dates >= val_start) & (dates <= val_end))
    if len(train_positions) == 0 or len(validation_positions) == 0:
        raise ValueError(
            f"empty design range: {len(train_positions)} train days, "
            f"{len(validation_positions)} validation days"
        )
    train = (int(train_positions[0]), int(train_positions[-1]) + 1 - horizon)
    validation = (int(validation_positions[0]), int(validation_positions[-1]) + 1)
    return DesignSplit(train=train, validation=validation)
