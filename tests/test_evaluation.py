"""Evaluation module tests; run directly: python3 tests/test_evaluation.py.

The data here is synthetic and only validates the formulas and the
pipeline, not results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.evaluation.dsr_pbo import (
    annualized_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    pbo_cscv,
)
from lq45.evaluation.metrics import (
    max_drawdown_and_duration,
    rank_ic_and_spread,
    summarize_series,
)
from lq45.evaluation.regimes import split_regimes
from lq45.evaluation.significance import (
    mean_difference_test,
    romano_wolf_stepdown,
    sharpe_difference_test_lw,
)


def test_basic_metrics() -> None:
    rng = np.random.default_rng(3)
    idx = pd.RangeIndex(252)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=252), index=idx)
    summary = summarize_series(returns)
    assert summary["n_days"] == 252
    assert -1.0 < summary["max_drawdown"] <= 0.0
    assert summary["cumulative_return"] > -1.0
    max_drawdown, duration = max_drawdown_and_duration(
        pd.Series(np.cumprod(1.0 + returns), index=idx)
    )
    assert max_drawdown <= 0.0 and duration >= 0
    assert abs(annualized_sharpe(returns)) < 5.0
    # Sortino uses the downside deviation over ALL periods, not the std of
    # the negative subset.
    downside = np.minimum(returns.to_numpy(), 0.0)
    vol_down = float(np.sqrt(np.mean(downside**2)))
    expected = float(np.sqrt(252) * returns.mean() / vol_down)
    assert abs(summary["sortino"] - expected) < 1e-9


def test_expected_max_sharpe_closed_form() -> None:
    gamma = 0.5772156649
    n, variance = 50, 0.5
    max_z = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n) + gamma * norm.ppf(
        1.0 - 1.0 / (n * np.e)
    )
    # sigma scales the WHOLE maxZ bracket (Bailey & Lopez de Prado 2014).
    assert abs(expected_max_sharpe(n, variance) - np.sqrt(variance) * max_z) < 1e-12
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(10, -1.0) == 0.0


def test_dsr_and_pbo() -> None:
    rng = np.random.default_rng(4)
    idx = pd.RangeIndex(500)
    returns = pd.Series(rng.normal(0.001, 0.01, size=500), index=idx)
    result = deflated_sharpe(returns, n_trials=50)
    assert 0.0 <= result["dsr"] <= 1.0
    assert result["benchmark"] >= 0.0
    matrix = pd.DataFrame(
        rng.normal(size=(400, 6)), columns=[f"s{i}" for i in range(6)]
    )
    pbo = pbo_cscv(matrix, n_groups=4)
    assert 0.0 <= pbo["pbo"] <= 1.0
    assert pbo["n_splits"] == 6.0


def test_dsr_uses_trial_variance() -> None:
    rng = np.random.default_rng(9)
    idx = pd.RangeIndex(500)
    returns = pd.Series(rng.normal(0.001, 0.01, size=500), index=idx)
    clustered = [0.05, 0.0501, 0.0499, 0.0502, 0.0498]
    tight = deflated_sharpe(returns, n_trials=5, trial_srs=clustered)
    assert tight["benchmark"] < 0.01  # SR0 ~ 0 when the trials are alike
    dispersed = [-0.5, -0.25, 0.0, 0.25, 0.5]
    wide = deflated_sharpe(returns, n_trials=5, trial_srs=dispersed)
    # Wide trial dispersion inflates SR0 and lowers DSR.
    assert wide["benchmark"] > tight["benchmark"]
    assert wide["dsr"] < tight["dsr"]


def test_significance() -> None:
    rng = np.random.default_rng(5)
    idx = pd.RangeIndex(600)
    returns_a = pd.Series(rng.normal(0.001, 0.01, size=600), index=idx)
    returns_b = pd.Series(rng.normal(0.0, 0.01, size=600), index=idx)
    lw = sharpe_difference_test_lw(returns_a, returns_b)
    assert 0.0 <= lw["p_value"] <= 1.0
    # Identical series have identical Sharpes -> no rejection.
    identical = sharpe_difference_test_lw(returns_a, returns_a)
    assert abs(identical["t_stat"]) < 1e-8
    assert identical["p_value"] > 0.99
    supplementary = mean_difference_test(returns_a, returns_b)
    assert 0.0 <= supplementary["p_value"] <= 1.0

    # Romano-Wolf: a strong persistent signal must be rejected, noise not.
    strong = pd.DataFrame({"s1": returns_b + 0.01, "s2": returns_b + 0.011})
    table = romano_wolf_stepdown(
        strong, returns_b, n_bootstrap=200, seed=0, block=21
    )
    assert {"strategy", "t_stat", "critical_value", "reject", "step"}.issubset(
        table.columns
    )
    assert bool(table["reject"].iloc[0])
    noise = pd.DataFrame(
        {"s1": returns_b, "s2": returns_b + pd.Series(rng.normal(0, 0.001, 600))}
    )
    table_noise = romano_wolf_stepdown(
        noise, returns_b, n_bootstrap=200, seed=0, block=21
    )
    assert not bool(table_noise["reject"].iloc[0])
    # Stepdown stops at the first non-rejection -> monotone rejects.
    reject = table["reject"].tolist()
    for i in range(len(reject) - 1):
        if not reject[i]:
            assert not reject[i + 1]
    assert table["step"].tolist() == list(range(1, len(table) + 1))


def test_regimes() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2021-06-01", "2019-06-01"]),
            "return": [0.01, 0.02, 0.03],
        }
    )
    regimes = {
        "covid": {"start": "2020-01-01", "end": "2020-12-31"},
        "recovery_rate_hike": {"start": "2021-01-01", "end": "2025-12-31"},
    }
    parts = split_regimes(frame, "date", regimes)
    assert len(parts["covid"]) == 1 and len(parts["recovery_rate_hike"]) == 1


def test_ranker_metric() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 6,
            "ticker": list("ABCDEF"),
            "pred_raw": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "true_raw": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    perfect = rank_ic_and_spread(frame)
    assert abs(perfect["rank_ic_mean"] - 1.0) < 1e-9
    assert perfect["top_minus_bottom_mean"] > 0
    assert perfect["n_dates"] == 1.0
    # A reversed ranking is perfectly negatively correlated.
    reversed_ic = rank_ic_and_spread(frame.assign(true_raw=-frame["true_raw"]))
    assert abs(reversed_ic["rank_ic_mean"] + 1.0) < 1e-9

    # Two dates of noisy rankings: IC stays inside [-1, 1], both dates counted.
    rng = np.random.default_rng(0)
    noisy = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 5 + ["2020-01-02"] * 5,
            "ticker": list("ABCDE") * 2,
            "pred_raw": rng.normal(size=10),
            "true_raw": rng.normal(size=10),
        }
    )
    result = rank_ic_and_spread(noisy)
    assert -1.0 <= result["rank_ic_mean"] <= 1.0
    assert result["n_dates"] == 2.0

    # A single 2-stock date is below the minimum and must be rejected.
    try:
        rank_ic_and_spread(frame.head(2))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no date has >= 3 stocks")


def main() -> int:
    test_basic_metrics()
    test_expected_max_sharpe_closed_form()
    test_dsr_and_pbo()
    test_dsr_uses_trial_variance()
    test_significance()
    test_regimes()
    test_ranker_metric()
    print("test_evaluation.py: 7 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
