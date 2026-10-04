"""Evaluation stage-5 module: metrics, DSR/PBO, significance, regimes."""

from lq45.evaluation.dsr_pbo import annualized_sharpe, deflated_sharpe, pbo_cscv
from lq45.evaluation.metrics import (
    daily_risk_free_rate,
    max_drawdown_and_duration,
    rank_ic_and_spread,
    summarize_series,
)
from lq45.evaluation.regimes import split_regimes
from lq45.evaluation.significance import (
    mean_difference_test,
    newey_west_covariance,
    romano_wolf_stepdown,
    sharpe_difference_test_lw,
)

__all__ = [
    "annualized_sharpe",
    "daily_risk_free_rate",
    "deflated_sharpe",
    "max_drawdown_and_duration",
    "mean_difference_test",
    "newey_west_covariance",
    "pbo_cscv",
    "rank_ic_and_spread",
    "romano_wolf_stepdown",
    "sharpe_difference_test_lw",
    "split_regimes",
    "summarize_series",
]
