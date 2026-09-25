from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / "data" / "daily"
WEEKLY_DIR = ROOT / "data" / "weekly"
REPORT_DIR = ROOT / "data" / "validation"

REQUIRED_OHLC = ["Open", "High", "Low", "Close"]
MAX_DAILY_GAP_DAYS = 10
MAX_WEEKLY_GAP_DAYS = 14
MAX_DAILY_STALE_DAYS = 5
MAX_WEEKLY_STALE_DAYS = 14


def market_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()


def validate_file(path: Path, frequency: str, expected_latest: pd.Timestamp | None = None) -> dict:
    result = {
        "file": path.name,
        "frequency": frequency,
        "status": "PASS",
        "rows": 0,
        "first_date": "",
        "last_date": "",
        "duplicate_dates": 0,
        "missing_close": 0,
        "invalid_ohlc": 0,
        "negative_values": 0,
        "return_mismatch": 0,
        "max_gap_days": 0,
        "gap_after": "",
        "stale_days": 0,
        "notes": "",
    }

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        result["rows"] = len(df)

        if df.empty:
            result["status"] = "FAIL"
            result["notes"] = "Empty file"
            return result

        if df["Date"].isna().any():
            result["status"] = "FAIL"
            result["notes"] = "Invalid/missing dates"
            return result

        df = df.sort_values("Date").reset_index(drop=True)
        result["first_date"] = df["Date"].min().date().isoformat()
        result["last_date"] = df["Date"].max().date().isoformat()

        result["duplicate_dates"] = int(df["Date"].duplicated().sum())

        if len(df) >= 2:
            date_gaps = df["Date"].diff().dt.days
            max_gap_position = date_gaps.idxmax()
            max_gap = int(date_gaps.loc[max_gap_position])
            result["max_gap_days"] = max_gap
            result["gap_after"] = (
                df.loc[max_gap_position - 1, "Date"].date().isoformat()
            )

        missing_cols = [c for c in REQUIRED_OHLC if c not in df.columns]
        if missing_cols:
            result["status"] = "FAIL"
            result["notes"] = f"Missing columns: {', '.join(missing_cols)}"
            return result

        for col in REQUIRED_OHLC:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        result["missing_close"] = int(df["Close"].isna().sum())
        result["invalid_ohlc"] = int(
            (
                (df["High"] < df["Low"])
                | (df["High"] < df["Open"])
                | (df["High"] < df["Close"])
                | (df["Low"] > df["Open"])
                | (df["Low"] > df["Close"])
            ).fillna(False).sum()
        )
        result["negative_values"] = int(
            (df[REQUIRED_OHLC] <= 0).any(axis=1).sum()
        )

        if frequency == "weekly":
            for period, col in [
                (1, "Return_1W"),
                (4, "Return_4W"),
                (12, "Return_12W"),
                (26, "Return_26W"),
                (52, "Return_52W"),
            ]:
                if col in df.columns:
                    expected = df["Close"].pct_change(period)
                    actual = pd.to_numeric(df[col], errors="coerce")
                    mismatch = (actual - expected).abs() > 1e-10
                    mismatch &= actual.notna() & expected.notna()
                    result["return_mismatch"] += int(mismatch.sum())

        reference = expected_latest if expected_latest is not None else market_today()
        latest = pd.Timestamp(df["Date"].max()).normalize()
        result["stale_days"] = max((reference - latest).days, 0)
        gap_limit = MAX_WEEKLY_GAP_DAYS if frequency == "weekly" else MAX_DAILY_GAP_DAYS
        stale_limit = MAX_WEEKLY_STALE_DAYS if frequency == "weekly" else MAX_DAILY_STALE_DAYS

        failed_checks = [
            f"{k}={result[k]}"
            for k in (
                "duplicate_dates",
                "missing_close",
                "invalid_ohlc",
                "negative_values",
                "return_mismatch",
            )
            if result[k]
        ]

        if result["max_gap_days"] > gap_limit:
            failed_checks.append(
                f"data_gap={result['max_gap_days']}d after {result['gap_after']}"
            )

        if result["stale_days"] > stale_limit:
            failed_checks.append(
                f"stale_data={result['stale_days']}d behind reference {reference.date()}"
            )

        if failed_checks:
            result["status"] = "FAIL"
            result["notes"] = "Failed: " + ", ".join(failed_checks)
        else:
            result["notes"] = "All checks passed"

    except Exception as exc:
        result["status"] = "FAIL"
        result["notes"] = f"Validation error: {exc}"

    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    latest_dates = []
    for path in sorted(DAILY_DIR.glob("*.csv")):
        try:
            dates = pd.to_datetime(pd.read_csv(path, usecols=["Date"])["Date"], errors="coerce")
            if dates.notna().any():
                latest_dates.append(dates.max().normalize())
        except Exception:
            pass
    reference = max(latest_dates) if latest_dates else market_today()

    for frequency, directory in [("daily", DAILY_DIR), ("weekly", WEEKLY_DIR)]:
        for path in sorted(directory.glob("*.csv")):
            results.append(validate_file(path, frequency, reference))

    report = pd.DataFrame(results)
    report_path = REPORT_DIR / "data_quality_report.csv"
    report.to_csv(report_path, index=False)

    failures = report[report["status"] != "PASS"]
    print("=" * 72)
    print("DATA VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Reference latest : {reference.date()}")
    print(f"Files checked : {len(report)}")
    print(f"PASS          : {(report['status'] == 'PASS').sum()}")
    print(f"FAIL          : {(report['status'] != 'PASS').sum()}")
    print(f"Report        : {report_path}")
    print("=" * 72)

    if not failures.empty:
        print("FAILED FILES:")
        print(
            failures[
                ["frequency", "file", "max_gap_days", "gap_after", "stale_days", "notes"]
            ].to_string(index=False)
        )
        raise SystemExit(1)

    print("ALL DATA VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
