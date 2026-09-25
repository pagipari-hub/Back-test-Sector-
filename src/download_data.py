from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sectors.json"
DATA_DIR = ROOT / "data" / "daily"
DEFAULT_START = "2005-01-01"
OVERLAP_DAYS = 90
MAX_STALE_DAYS = 5
NSE_CHUNK_DAYS = 180
NSE_RETRIES = 3

NSE_INDEX_NAMES = {
    "NIFTY 50": "NIFTY 50", "NIFTY AUTO": "NIFTY AUTO",
    "NIFTY BANK": "NIFTY BANK", "NIFTY IT": "NIFTY IT",
    "NIFTY PHARMA": "NIFTY PHARMA", "NIFTY FMCG": "NIFTY FMCG",
    "NIFTY METAL": "NIFTY METAL", "NIFTY REALTY": "NIFTY REALTY",
    "NIFTY PSU BANK": "NIFTY PSU BANK", "NIFTY MEDIA": "NIFTY MEDIA",
    "NIFTY ENERGY": "NIFTY ENERGY",
}


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



def yahoo_download(ticker: str, start: str) -> pd.DataFrame:
    return normalize(yf.download(ticker, start=start, auto_adjust=False,
                                 progress=False, actions=False, threads=False))


def nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    session.get("https://www.nseindia.com/", timeout=20)
    return session


def nse_fetch_chunk(session: requests.Session, index_name: str,
                    start: date, end: date) -> pd.DataFrame:
    url = "https://www.nseindia.com/api/historical/indicesHistory"
    params = {"indexType": index_name, "from": start.strftime("%d-%m-%Y"),
              "to": end.strftime("%d-%m-%Y")}
    last_error = None
    for attempt in range(1, NSE_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            rows = r.json().get("data", [])
            if not rows:
                return pd.DataFrame()
            frame = pd.DataFrame(rows)
            date_col = next((c for c in ("TIMESTAMP", "Date", "date") if c in frame), None)
            aliases = {"Open": ("OPEN_INDEX_VAL", "OPEN"),
                       "High": ("HIGH_INDEX_VAL", "HIGH"),
                       "Low": ("LOW_INDEX_VAL", "LOW"),
                       "Close": ("CLOSE_INDEX_VAL", "CLOSE")}
            if date_col is None:
                raise ValueError("NSE response has no date column")
            out = pd.DataFrame(index=pd.to_datetime(frame[date_col], dayfirst=True))
            for target, candidates in aliases.items():
                source = next((c for c in candidates if c in frame), None)
                if source is None:
                    raise ValueError(f"NSE response missing {target}")
                out[target] = pd.to_numeric(frame[source], errors="coerce")
            out.index = out.index.tz_localize(None)
            out.index.name = "Date"
            return out.dropna(subset=["Close"])
        except Exception as exc:
            last_error = exc
            if attempt < NSE_RETRIES:
                time.sleep(attempt * 2)
    raise RuntimeError(f"NSE fetch failed for {index_name}: {last_error}")


def nse_download(index_name: str, start: str) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="Asia/Kolkata").date()
    cursor = pd.Timestamp(start).date()
    session = nse_session()
    chunks = []
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=NSE_CHUNK_DAYS).to_pytimedelta(), end)
        print(f"  NSE fallback: {index_name} {cursor} -> {chunk_end}")
        chunks.append(nse_fetch_chunk(session, index_name, cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]


def is_stale(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    latest = pd.Timestamp(df.index.max()).normalize()
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    return (today - latest).days > MAX_STALE_DAYS


def merge_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    frames = [x for x in frames if not x.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, sort=False)
    return combined[~combined.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])


def download_series(name: str, ticker: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / f"{name.lower().replace(' ', '_')}.csv"
    existing = read_existing(output)
    if existing.empty:
        start = DEFAULT_START
    else:
        start = (existing.index.max() - pd.Timedelta(days=OVERLAP_DAYS)).date().isoformat()

    print(f"Updating {name} [{ticker}] from {start}...")
    try:
        yahoo = yahoo_download(ticker, start)
    except Exception as exc:
        print(f"WARNING: Yahoo download failed for {ticker}: {exc}")
        yahoo = pd.DataFrame()

    combined = merge_frames(existing, yahoo)
    source = "Yahoo"

    if is_stale(combined):
        index_name = NSE_INDEX_NAMES.get(name)
        if not index_name:
            raise RuntimeError(f"No NSE fallback mapping configured for {name}")
        fallback_start = (
            (combined.index.max() - pd.Timedelta(days=OVERLAP_DAYS)).date().isoformat()
            if not combined.empty else DEFAULT_START
        )
        latest = combined.index.max().date().isoformat() if not combined.empty else "none"
        print(f"WARNING: {name} is stale at {latest}; fetching NSE tail from {fallback_start}")
        nse = nse_download(index_name, fallback_start)
        combined = merge_frames(combined, nse)
        source = "Yahoo + NSE"

    if combined.empty or is_stale(combined):
        latest = combined.index.max().date().isoformat() if not combined.empty else "none"
        raise RuntimeError(f"{name} remains stale after fallback: latest={latest}")

    combined.index.name = "Date"
    combined.to_csv(output)
    print(f"Stored {len(combined):,} rows -> {output} [source: {source}; latest: {combined.index.max().date()}]")

def main() -> None:
    config = load_config()
    download_series(config["benchmark"]["name"], config["benchmark"]["ticker"])
    for name, ticker in config["sectors"].items():
        download_series(name, ticker)


if __name__ == "__main__":
    main()
