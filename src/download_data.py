from __future__ import annotations

import json
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sectors.json"
DATA_DIR = ROOT / "data" / "daily"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def download_series(name: str, ticker: str, start: str = "2005-01-01") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} [{ticker}]...")

    df = yf.download(
        ticker,
        start=start,
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if df.empty:
        print(f"  WARNING: no data returned for {ticker}")
        return

    if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    output = DATA_DIR / f"{name.lower().replace(' ', '_')}.csv"
    df.to_csv(output)
    print(f"  {len(df):,} rows -> {output}")


def main() -> None:
    config = load_config()

    download_series(
        config["benchmark"]["name"],
        config["benchmark"]["ticker"],
    )

    for name, ticker in config["sectors"].items():
        download_series(name, ticker)


if __name__ == "__main__":
    main()
