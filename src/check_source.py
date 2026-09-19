from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "sectors.json"
START_DATE = "2026-07-01"


def max_gap_days(index: pd.DatetimeIndex) -> tuple[int, str]:
    if len(index) < 2:
        return 0, ""

    dates = pd.Series(index).sort_values().reset_index(drop=True)
    gaps = dates.diff().dt.days
    position = gaps.idxmax()

    if pd.isna(position):
        return 0, ""

    return int(gaps.loc[position]), dates.loc[position - 1].date().isoformat()


def download_dates(ticker: str) -> pd.DatetimeIndex:
    df = yf.download(
        ticker,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
        actions=False,
    )

    if df.empty:
        return pd.DatetimeIndex([])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    dates = pd.to_datetime(df.index, errors="coerce")
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    dates = pd.DatetimeIndex(dates).dropna().normalize().drop_duplicates().sort_values()
    return dates


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    tickers = [("NIFTY 50", config["benchmark"]["ticker"])]
    tickers.extend(config["sectors"].items())

    print(f"Source check: yfinance from {START_DATE}")
    print("=" * 100)

    for name, ticker in tickers:
        try:
            dates = download_dates(ticker)
            if len(dates) == 0:
                print(
                    f"{name:15} {ticker:12} rows=0 first=NA last=NA max_gap=0d gap_after=NA"
                )
                continue

            gap, after = max_gap_days(dates)
            print(
                f"{name:15} {ticker:12} rows={len(dates):4d} "
                f"first={dates[0].date()} last={dates[-1].date()} "
                f"max_gap={gap:2d}d gap_after={after or 'NA'}"
            )
        except Exception as exc:
            print(f"{name:15} {ticker:12} ERROR: {exc}")


if __name__ == "__main__":
    main()
