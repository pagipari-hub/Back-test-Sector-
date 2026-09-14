from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import buy_and_hold, performance_metrics


def test_buy_and_hold_known_result():
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    prices = pd.DataFrame({"Close": [100.0, 110.0, 121.0, 133.1]}, index=dates)

    trades, equity, metrics = buy_and_hold(prices, "TEST")

    assert len(trades) == 3
    assert math.isclose(metrics["Final_Capital"], 133100.0, rel_tol=1e-12)
    assert math.isclose(metrics["Total_Return"], 0.331, rel_tol=1e-12)


def test_drawdown_is_zero_for_monotonic_growth():
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    equity = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates)
    returns = equity.pct_change()

    metrics = performance_metrics(equity, returns, pd.DataFrame({"Return": returns.dropna()}))

    assert math.isclose(metrics["Max_Drawdown"], 0.0, abs_tol=1e-12)


def test_known_drawdown():
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    equity = pd.Series([100.0, 120.0, 90.0, 99.0], index=dates)
    returns = equity.pct_change()

    metrics = performance_metrics(equity, returns, pd.DataFrame({"Return": returns.dropna()}))

    assert math.isclose(metrics["Max_Drawdown"], -0.25, rel_tol=1e-12)
