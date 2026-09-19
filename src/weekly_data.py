from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / "data" / "daily"
WEEKLY_DIR = ROOT / "data" / "weekly"


def to_weekly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    numeric = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    weekly = pd.DataFrame(index=df.resample("W-FRI").last().index)
    weekly["Open"] = df["Open"].resample("W-FRI").first()
    weekly["High"] = df["High"].resample("W-FRI").max()
    weekly["Low"] = df["Low"].resample("W-FRI").min()
    weekly["Close"] = df["Close"].resample("W-FRI").last()
    if "Adj Close" in df:
        weekly["Adj Close"] = df["Adj Close"].resample("W-FRI").last()
    if "Volume" in df:
        weekly["Volume"] = df["Volume"].resample("W-FRI").sum(min_count=1)

    weekly = weekly.dropna(subset=["Close"])

    weekly["Return_1W"] = weekly["Close"].pct_change()
    weekly["Return_4W"] = weekly["Close"].pct_change(4)
    weekly["Return_12W"] = weekly["Close"].pct_change(12)
    weekly["Return_26W"] = weekly["Close"].pct_change(26)
    weekly["Return_52W"] = weekly["Close"].pct_change(52)
    return weekly


def main() -> None:
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(DAILY_DIR.glob("*.csv")):
        weekly = to_weekly(path)
        output = WEEKLY_DIR / path.name
        weekly.to_csv(output)
        print(f"{path.stem}: {len(weekly):,} weekly rows -> {output}")


if __name__ == "__main__":
    main()
