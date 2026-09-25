#!/usr/bin/env python3
"""
Fetch missing data from Yahoo Finance and update daily/weekly CSV files.

Handles:
- 7 NIFTY sectors with 63-day data gap (ending ~2026-07-17)
- Daily data fetch + weekly resampling (W-FRI, last close)
- Schema preservation, deduplication, rate limiting
"""

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Sector ticker mappings
SECTOR_TICKERS = {
    "nifty_auto": "^CNXAUTO",
    "nifty_energy": "^CNXENERGY",
    "nifty_fmcg": "^CNXFMCG",
    "nifty_media": "^CNXMEDIA",
    "nifty_metal": "^CNXMETAL",
    "nifty_psu_bank": "^CNXPSUBANK",
    "nifty_realty": "^CNXREALTY",
}

ROOT = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / "data" / "daily"
WEEKLY_DIR = ROOT / "data" / "weekly"

DAILY_COLUMNS = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
WEEKLY_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Return_1W",
    "Return_4W",
    "Return_12W",
    "Return_26W",
    "Return_52W",
]


def fetch_sector_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV data from Yahoo Finance for a sector ticker.

    Returns None if fetch fails, otherwise returns DataFrame with 'Date' column.
    """
    try:
        print(f"  Fetching {ticker} from {start_date} to {end_date}...", end=" ")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            print("NO DATA")
            return None

        # Handle MultiIndex columns (from newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index to convert Date from index to column
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()

        # Rename columns to match expected schema
        col_map = {
            "Adj Close": "Adj Close",
            "Close": "Close",
            "High": "High",
            "Low": "Low",
            "Open": "Open",
            "Volume": "Volume",
        }
        df = df.rename(columns=col_map)

        # Keep only needed columns
        df = df[["Date"] + [c for c in col_map.keys() if c in df.columns]]

        df = df.sort_values("Date").reset_index(drop=True)
        print(f"{len(df)} rows")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def update_daily_csv(sector_name: str, ticker: str) -> tuple[str, str, int]:
    """
    Update daily CSV for a sector.

    Returns (sector_name, file_path, rows_added) or raises on error.
    """
    csv_path = DAILY_DIR / f"{sector_name}.csv"

    # Read existing
    existing = pd.read_csv(csv_path, parse_dates=["Date"])
    existing["Date"] = existing["Date"].dt.normalize()
    existing = existing.sort_values("Date").reset_index(drop=True)

    last_date = existing["Date"].max()
    last_date_str = last_date.strftime("%Y-%m-%d")
    fetch_start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end_str = "2026-09-26"  # One day ahead to ensure today is included

    print(f"  Last date in {sector_name}: {last_date_str}")

    # Fetch new data
    new_data = fetch_sector_data(ticker, fetch_start_str, fetch_end_str)
    if new_data is None or new_data.empty:
        print(f"  WARNING: No new data fetched for {sector_name}")
        return (sector_name, str(csv_path), 0)

    rows_added = len(new_data)
    print(f"  Fetched {rows_added} new rows")

    # Merge: concat old + new, drop duplicates keeping latest, sort
    merged = pd.concat([existing, new_data], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="last")
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Preserve schema: ensure exact columns in order
    for col in DAILY_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0 if col == "Volume" else 0.0

    merged = merged[DAILY_COLUMNS]

    # Format date and numeric columns
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    for col in ["Adj Close", "Close", "High", "Low", "Open"]:
        merged[col] = merged[col].astype(float)
    merged["Volume"] = merged["Volume"].astype(int)

    # Write
    merged.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path}: now {len(merged)} rows total")

    return (sector_name, str(csv_path), rows_added)


def update_weekly_csv(sector_name: str) -> tuple[str, str, int]:
    """
    Update weekly CSV by resampling from daily data.
    Resamples to W-FRI (Friday), uses last Close value per week.
    """
    daily_path = DAILY_DIR / f"{sector_name}.csv"
    weekly_path = WEEKLY_DIR / f"{sector_name}.csv"

    # Read daily
    daily_df = pd.read_csv(daily_path, parse_dates=["Date"])
    daily_df["Date"] = daily_df["Date"].dt.normalize()

    # Read existing weekly to extract return columns (if needed to preserve indices)
    existing_weekly = pd.read_csv(weekly_path, parse_dates=["Date"])
    existing_weekly["Date"] = existing_weekly["Date"].dt.normalize()
    existing_weekly = existing_weekly.sort_values("Date").reset_index(drop=True)

    last_weekly_date = existing_weekly["Date"].max()
    print(f"  Last weekly date in {sector_name}: {last_weekly_date.strftime('%Y-%m-%d')}")

    # Resample daily to weekly (Friday close)
    weekly_resampled = (
        daily_df.set_index("Date")
        .resample("W-FRI")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Adj Close": "last",
            "Volume": "sum",
        })
    )
    weekly_resampled = weekly_resampled.reset_index()
    weekly_resampled = weekly_resampled[weekly_resampled["Date"].notna()]

    # Merge: keep old weeks, append/overwrite weeks >= last_weekly_date
    old_weeks = existing_weekly[existing_weekly["Date"] < last_weekly_date].copy()
    new_weeks = weekly_resampled[weekly_resampled["Date"] >= last_weekly_date].copy()

    merged_weekly = pd.concat([old_weeks, new_weeks], ignore_index=True)
    merged_weekly = merged_weekly.drop_duplicates(subset=["Date"], keep="last")
    merged_weekly = merged_weekly.sort_values("Date").reset_index(drop=True)

    # Recalculate returns
    merged_weekly["Return_1W"] = merged_weekly["Close"].pct_change(1)
    merged_weekly["Return_4W"] = merged_weekly["Close"].pct_change(4)
    merged_weekly["Return_12W"] = merged_weekly["Close"].pct_change(12)
    merged_weekly["Return_26W"] = merged_weekly["Close"].pct_change(26)
    merged_weekly["Return_52W"] = merged_weekly["Close"].pct_change(52)

    # Preserve schema
    for col in WEEKLY_COLUMNS:
        if col not in merged_weekly.columns:
            merged_weekly[col] = 0.0

    merged_weekly = merged_weekly[WEEKLY_COLUMNS]

    # Format
    merged_weekly["Date"] = merged_weekly["Date"].dt.strftime("%Y-%m-%d")
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        merged_weekly[col] = merged_weekly[col].astype(float)
    merged_weekly["Volume"] = merged_weekly["Volume"].astype(float)
    for col in [
        "Return_1W",
        "Return_4W",
        "Return_12W",
        "Return_26W",
        "Return_52W",
    ]:
        merged_weekly[col] = merged_weekly[col].astype(float)

    rows_added = len(new_weeks)
    merged_weekly.to_csv(weekly_path, index=False)
    print(f"  Updated {weekly_path}: now {len(merged_weekly)} rows total")

    return (sector_name, str(weekly_path), rows_added)


def main():
    print("=" * 72)
    print("UPDATE SECTOR DATA FROM YAHOO FINANCE")
    print("=" * 72)

    summary = []

    for sector_name, ticker in SECTOR_TICKERS.items():
        print(f"\nProcessing {sector_name} ({ticker})...")

        # Update daily
        try:
            result_daily = update_daily_csv(sector_name, ticker)
            summary.append(("daily", *result_daily))
        except Exception as e:
            print(f"  FAILED to update daily: {e}")
            summary.append(("daily", sector_name, str("ERROR"), -1))

        # Rate limit
        time.sleep(1)

        # Update weekly (resample from daily)
        try:
            result_weekly = update_weekly_csv(sector_name)
            summary.append(("weekly", *result_weekly))
        except Exception as e:
            print(f"  FAILED to update weekly: {e}")
            summary.append(("weekly", sector_name, str("ERROR"), -1))

        # Rate limit
        time.sleep(1)

    # Print summary
    print("\n" + "=" * 72)
    print("UPDATE SUMMARY")
    print("=" * 72)
    for freq, sector, filepath, rows_added in summary:
        status = f"+{rows_added}" if rows_added >= 0 else "ERROR"
        print(f"{freq:6s} | {sector:20s} | {status:>5s} rows")

    print("=" * 72)
    print("Run 'python src/validate_data.py' to verify all files")
    print("=" * 72)


if __name__ == "__main__":
    main()
