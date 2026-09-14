from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEEKLY_DIR = ROOT / "data" / "weekly"
OUTPUT_DIR = ROOT / "backtest"

INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.0


def load_weekly(filename: str) -> pd.DataFrame:
    df = pd.read_csv(WEEKLY_DIR / filename, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"])


def performance_metrics(equity: pd.Series, weekly_returns: pd.Series, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    weekly_returns = weekly_returns.dropna()
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = end_value / start_value - 1
    cagr = (end_value / start_value) ** (1 / years) - 1
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    vol = float(weekly_returns.std(ddof=1) * math.sqrt(52)) if len(weekly_returns) > 1 else 0.0
    annual_return = float(weekly_returns.mean() * 52) if len(weekly_returns) else 0.0
    sharpe = (annual_return - RISK_FREE_RATE) / vol if vol > 0 else 0.0
    win_rate = float((trades["Return"] > 0).mean()) if not trades.empty else 0.0

    return {
        "Start": equity.index[0].date().isoformat(),
        "End": equity.index[-1].date().isoformat(),
        "Initial_Capital": start_value,
        "Final_Capital": end_value,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Max_Drawdown": max_drawdown,
        "Annualized_Volatility": vol,
        "Sharpe": sharpe,
        "Trades": int(len(trades)),
        "Win_Rate": win_rate,
    }


def buy_and_hold(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    prices = df["Close"].copy()
    returns = prices.pct_change().fillna(0.0)
    equity = INITIAL_CAPITAL * (1 + returns).cumprod()

    trades = pd.DataFrame({
        "Date": prices.index[1:],
        "Asset": name,
        "Return": returns.iloc[1:].values,
    })
    metrics = performance_metrics(equity, returns, trades)
    return trades, equity, metrics


def sector_rotation(sector_files: list[str], lookback_weeks: int = 12) -> tuple[pd.DataFrame, pd.Series, dict]:
    frames = {}
    for filename in sector_files:
        name = Path(filename).stem
        df = load_weekly(filename)
        frames[name] = df["Close"].rename(name)

    closes = pd.concat(frames.values(), axis=1).sort_index()
    roc = closes / closes.shift(lookback_weeks) - 1
    forward_return = closes.shift(-1) / closes - 1

    rows = []
    for date in roc.index:
        scores = roc.loc[date].dropna()
        if scores.empty or date == roc.index[-1]:
            continue
        selected = scores.idxmax()
        next_return = forward_return.loc[date, selected]
        if pd.isna(next_return):
            continue
        rows.append({
            "Signal_Date": date,
            "Asset": selected,
            "ROC": float(scores[selected]),
            "Return": float(next_return),
        })

    trades = pd.DataFrame(rows)
    if trades.empty:
        raise RuntimeError("No sector rotation trades could be generated")

    trades["Date"] = pd.to_datetime(trades["Signal_Date"])
    trades = trades.sort_values("Date").reset_index(drop=True)
    weekly_returns = trades.set_index("Date")["Return"]
    equity = INITIAL_CAPITAL * (1 + weekly_returns).cumprod()
    metrics = performance_metrics(equity, weekly_returns, trades)
    return trades, equity, metrics


def save_result(name: str, trades: pd.DataFrame, equity: pd.Series, metrics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUTPUT_DIR / f"{name}_trades.csv", index=False)
    equity.rename("Equity").to_csv(OUTPUT_DIR / f"{name}_equity.csv", header=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / f"{name}_performance.csv", index=False)


def main() -> None:
    config = json.loads((ROOT / "config" / "sectors.json").read_text(encoding="utf-8"))
    benchmark_file = "nifty_50.csv"
    sector_files = [f"{name.lower().replace(' ', '_')}.csv" for name in config["sectors"]]

    print("BACKTEST SANITY SUITE")
    print("-" * 72)

    benchmark_trades, benchmark_equity, benchmark_metrics = buy_and_hold(
        load_weekly(benchmark_file), "NIFTY 50"
    )
    save_result("nifty50_buy_hold", benchmark_trades, benchmark_equity, benchmark_metrics)

    rotation_trades, rotation_equity, rotation_metrics = sector_rotation(sector_files, lookback_weeks=12)
    save_result("sector_rotation_12w_roc", rotation_trades, rotation_equity, rotation_metrics)

    report = pd.DataFrame([
        {"Strategy": "NIFTY 50 Buy & Hold", **benchmark_metrics},
        {"Strategy": "Sector Rotation: strongest 12W ROC", **rotation_metrics},
    ])
    report.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)

    with pd.ExcelWriter(OUTPUT_DIR / "backtest_report.xlsx", engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Performance", index=False)
        benchmark_trades.to_excel(writer, sheet_name="NIFTY50_Trades", index=False)
        rotation_trades.to_excel(writer, sheet_name="Rotation_Trades", index=False)

    print(report.to_string(index=False))
    print("-" * 72)
    print("BACKTEST ENGINE COMPLETED")
    print(f"Reports: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
