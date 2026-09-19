from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sectors.json"
DATA_DIR = ROOT / "data" / "daily"
DEFAULT_START = "2005-01-01"
OVERLAP_DAYS = 90


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def read_existing(output: Path) -> pd.DataFrame:
    if not output.exists():
        return pd.DataFrame()
    df = pd.read_csv(output, parse_dates=["Date"])
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")


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
        start = (last_date - pd.Timedelta(days=OVERLAP_DAYS)).date().isoformat()
        print(f"Refreshing {name} [{ticker}] from {start} (last stored: {last_date.date()})...")

    new = normalize(
        yf.download(
            ticker,
            start=start,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )
    )

    if new.empty:
        if existing.empty:
            print(f"WARNING: no data returned for {ticker}")
        else:
            print("No rows returned; existing data retained.")
        return

    combined = new if existing.empty else pd.concat([existing, new])
    combined = combined.dropna(subset=["Close"])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index.name = "Date"
    combined.to_csv(output)
    added = max(len(combined) - len(existing), 0)
    print(f"Stored {len(combined):,} rows (+{added:,} new) -> {output}")


def main() -> None:
    config = load_config()
    download_series(config["benchmark"]["name"], config["benchmark"]["ticker"])
    for name, ticker in config["sectors"].items():
        download_series(name, ticker)


if __name__ == "__main__":
    main()
