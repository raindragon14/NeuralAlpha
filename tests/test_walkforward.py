"""Walk-forward fold tests; run directly: python3 tests/test_walkforward.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.walkforward import design_split, make_folds

SPLIT = {
    "initial_train_years": 2,
    "validation_days": 63,
    "test_days": 21,
    "step_days": 21,
    "purge": {"horizon_days": 5},
    "embargo_days": 59,
}


def test_count_and_bounds() -> None:
    folds = make_folds(1932, SPLIT)
    assert len(folds) == 65
    first = folds[0]
    assert first.train == (0, 499)  # 504 - purge 5
    assert first.validation == (504, 567)
    assert first.test == (567, 588)
    assert first.banned == []
    last = folds[-1]
    assert last.test[1] == 1932  # covers the calendar exactly


def test_embargo() -> None:
    folds = make_folds(1932, SPLIT)
    # Fold 0's test ends at 588; the ban [588, 647) shows up in every
    # following fold.
    for fold in folds[1:]:
        assert (588, 647) in [tuple(b) for b in fold.banned]


def test_purge_and_embargo_params() -> None:
    folds = make_folds(1932, SPLIT, purge_days=21)
    assert folds[0].train == (0, 483)  # 504 - 21
    folds = make_folds(1932, SPLIT, embargo_days=29)
    for fold in folds[1:]:
        assert (588, 617) in [tuple(b) for b in fold.banned]


def test_test_covers_oos() -> None:
    folds = make_folds(1932, SPLIT)
    covered = np.zeros(1932, dtype=bool)
    for fold in folds:
        start, end = fold.test
        assert not covered[start:end].any()  # test folds do not overlap
        covered[start:end] = True
    assert covered[:567].sum() == 0  # 63 opening OOS days carry no test
    assert covered[567:].all()  # the remainder is fully covered


def test_design_split() -> None:
    dates = pd.date_range("2018-01-01", "2019-12-31", freq="B")
    design = design_split(
        dates, "2018-01-01/2018-12-31", "2019-01-01/2019-12-31", horizon=5
    )
    pos_2018 = np.flatnonzero(dates.year == 2018)
    pos_2019 = np.flatnonzero(dates.year == 2019)
    assert design.train == (int(pos_2018[0]), int(pos_2018[-1]) + 1 - 5)
    assert design.validation == (int(pos_2019[0]), int(pos_2019[-1]) + 1)


def main() -> int:
    test_count_and_bounds()
    test_embargo()
    test_purge_and_embargo_params()
    test_test_covers_oos()
    test_design_split()
    print("test_walkforward.py: 5 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
