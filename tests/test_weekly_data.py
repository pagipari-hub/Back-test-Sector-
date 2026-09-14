from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weekly_data import to_weekly


def test_weekly_ohlc_aggregation(tmp_path):
    dates = pd.to_datetime([
        "2025-01-06", "2025-01-07", "2025-01-10",
        "2025-01-13", "2025-01-17",
    ])
    source = pd.DataFrame({
        "Open": [100, 102, 101, 110, 112],
        "High": [103, 105, 106, 115, 118],
        "Low": [99, 100, 98, 109, 111],
        "Close": [102, 104, 105, 113, 117],
        "Volume": [10, 20, 30, 40, 50],
    }, index=dates)
    source.index.name = "Date"

    path = tmp_path / "test.csv"
    source.to_csv(path)

    weekly = to_weekly(path)

    first = weekly.iloc[0]
    assert first["Open"] == 100
    assert first["High"] == 106
    assert first["Low"] == 98
    assert first["Close"] == 105
    assert first["Volume"] == 60
