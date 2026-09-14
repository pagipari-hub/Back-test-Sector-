from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sectors.json"
DATA_DIR = ROOT / "data" / "daily"
DEFAULT_START = "2005-01-01"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def read_existing(output: Path) -> pd.DataFrame:
    if not output.exists():
        return pd.DataFrame()
    df = pd.read_csv(output, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df = df.set_index("Date")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def download_series(name: str, ticker: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / f"{name.lower().replace(' ', '_')}.csv"
    existing = read_existing(output)

    if existing.empty:
        start = DEFAULT_START
        print(f"Downloading {name} [{ticker}] from {start}...")
    else:
        last_date = existing.index.max()
        start = (last_date + pd.Timedelta(days=1)).date().isoformat()
        print(f"Updating {name} [{ticker}] from {start} (last stored: {last_date.date()})...")

    new = normalize(yf.download(
        ticker,
        start=start,
        auto_adjust=False,
        progress=False,
        actions=False,
    ))

    if new.empty:
        if existing.empty:
            print(f"  WARNING: no data returned for {ticker}")
        else:
            print("  Up to date: no new rows.")
        return

    if existing.empty:
        combined = new
    else:
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    combined.index.name = "Date"
    combined.to_csv(output)
    added = len(combined) - len(existing)
    print(f"  Stored {len(combined):,} rows (+{max(added, 0):,} new) -> {output}")


def main() -> None:
    config = load_config()
    download_series(config["benchmark"]["name"], config["benchmark"]["ticker"])
    for name, ticker in config["sectors"].items():
        download_series(name, ticker)


if __name__ == "__main__":
    main()
