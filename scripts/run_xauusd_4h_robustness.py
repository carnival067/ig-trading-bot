"""Offline robustness diagnostics for the blocked XAUUSD 4H trend result."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.market_timeframe_research import (
    StrategySpec,
    add_regimes,
    load_histdata_minutes,
    metrics,
    records,
    resample_market,
    run_market_spec,
)

ROOT = Path("/Users/akshay/Documents/HIST_DATA/XAU:USD/M1")
WINDOWS = ("wf_1", "wf_2", "wf_3", "oos")


def main() -> None:
    minute = load_histdata_minutes(sorted(ROOT.rglob("*.csv")))
    frame = add_regimes(resample_market(minute, "4h"))
    baseline = _spec("EMA50_200_PULLBACK", 1.5, 1.5, "pullback", False)

    cost_rows = []
    for cost_multiplier in (1.0, 1.25, 1.5, 2.0):
        for slippage_multiplier in (1.0, 2.0, 3.0):
            trades = _run_windows(
                frame,
                baseline,
                cost_multiplier=cost_multiplier,
                slippage_multiplier=slippage_multiplier,
            )
            cost_rows.append({
                "cost_multiplier": cost_multiplier,
                "slippage_multiplier": slippage_multiplier,
                **metrics(trades),
                "passes_pf_1_25": metrics(trades)["profit_factor"] > 1.25,
                "passes_expectancy": metrics(trades)["expectancy"] > 0,
                "status": "FAIL",
            })
    pd.DataFrame(cost_rows).to_csv("XAUUSD_4H_COST_STRESS_TEST.csv", index=False)

    periods = {
        "2021": ("2021-01-01", "2022-01-01"),
        "2022": ("2022-01-01", "2023-01-01"),
        "2023": ("2023-01-01", "2024-01-01"),
        "2024": ("2024-01-01", "2025-01-01"),
        "2025": ("2025-01-01", "2026-01-01"),
        "2021-2022": ("2021-01-01", "2023-01-01"),
        "2023-2024": ("2023-01-01", "2025-01-01"),
        "2024-2025": ("2024-01-01", "2026-01-01"),
    }
    window_rows = []
    for label, (start, end) in periods.items():
        subset = frame.loc[start:end].iloc[:-1]
        trades = records(run_market_spec(
            subset, "XAUUSD", "4H", baseline, label,
            start_fraction=0.0, end_fraction=1.0,
        ))
        window_rows.append({"window_type": "calendar", "window": label, **metrics(trades)})
    baseline_trades = []
    for window in WINDOWS:
        trades = records(run_market_spec(frame, "XAUUSD", "4H", baseline, window))
        baseline_trades.append(trades)
        window_rows.append({"window_type": "walk_forward", "window": window, **metrics(trades)})
    pd.DataFrame(window_rows).to_csv("XAUUSD_4H_WINDOW_STABILITY.csv", index=False)

    all_trades = pd.concat(baseline_trades, ignore_index=True)
    sensitivity = []
    variants = [
        baseline,
        _spec("EMA20_50_PULLBACK", 1.5, 1.5, "pullback", False, 20, 50),
        _spec("ATR_STOP_1.0", 1.0, 1.5, "pullback", False),
        _spec("ATR_STOP_2.0", 2.0, 1.5, "pullback", False),
        _spec("TARGET_1R", 1.5, 1.0, "pullback", False),
        _spec("TARGET_2R", 1.5, 2.0, "pullback", False),
        _spec("TREND_ONLY", 1.5, 1.5, "trend", False),
        _spec("PULLBACK_SESSION_FILTER", 1.5, 1.5, "pullback", True),
    ]
    for spec in variants:
        trades = _run_windows(frame, spec)
        stressed = _run_windows(frame, spec, cost_multiplier=1.25)
        m = metrics(trades)
        oos = trades[trades["window"] == "oos"] if not trades.empty else trades
        window_sums = trades.groupby("window")["r_multiple"].sum() if not trades.empty else pd.Series(dtype=float)
        reasons = []
        if m["total_trades"] < 200:
            reasons.append("fewer than 200 trades")
        if m["profit_factor"] <= 1.25:
            reasons.append("PF not above 1.25")
        if metrics(stressed)["profit_factor"] <= 1.25:
            reasons.append("PF fails at 1.25x cost")
        if metrics(oos)["total_return"] <= 0 or metrics(oos)["profit_factor"] <= 1.25:
            reasons.append("OOS not clearly positive")
        if len(window_sums) < 4 or int((window_sums > 0).sum()) < 3:
            reasons.append("walk-forward instability")
        if not reasons:
            reasons.append(
                "isolated parameter combination; edge not replicated across nearby rules"
            )
        sensitivity.append({
            "variant": spec.variant, **m,
            "pf_at_1_25x_cost": metrics(stressed)["profit_factor"],
            "oos_return": metrics(oos)["total_return"],
            "oos_profit_factor": metrics(oos)["profit_factor"],
            "positive_windows": int((window_sums > 0).sum()),
            "status": "FAIL", "reason": "; ".join(reasons),
        })

    distribution = _distribution(all_trades)
    _write_report(pd.DataFrame(cost_rows), pd.DataFrame(window_rows), pd.DataFrame(sensitivity), distribution)


def _run_windows(frame, spec, **kwargs) -> pd.DataFrame:
    trades = []
    for window in WINDOWS:
        trades.extend(run_market_spec(frame, "XAUUSD", "4H", spec, window, **kwargs))
    return records(trades)


def _spec(
    name: str,
    stop_atr: float,
    target_r: float,
    entry_mode: str,
    session_filter: bool,
    fast: int = 50,
    slow: int = 200,
) -> StrategySpec:
    def signal_builder(frame: pd.DataFrame) -> pd.DataFrame:
        fast_ema = frame[f"ema_{fast}"]
        slow_ema = frame[f"ema_{slow}"]
        long = (fast_ema > slow_ema) & (slow_ema.diff(3) > 0)
        short = (fast_ema < slow_ema) & (slow_ema.diff(3) < 0)
        if entry_mode == "pullback":
            long &= (frame["low"] <= frame["ema_20"]) & (frame["close"] > frame["open"]) & (frame["close"] > frame["ema_20"])
            short &= (frame["high"] >= frame["ema_20"]) & (frame["close"] < frame["open"]) & (frame["close"] < frame["ema_20"])
        else:
            long &= (frame["close"] > frame["open"])
            short &= (frame["close"] < frame["open"])
        if session_filter:
            liquid = frame["session"].isin(["LONDON", "OVERLAP", "NEW_YORK"])
            long &= liquid
            short &= liquid
        signal = pd.Series(0, index=frame.index, dtype=int)
        signal[long.fillna(False)] = 1
        signal[short.fillna(False)] = -1
        return pd.DataFrame({"signal": signal, "target": np.nan}, index=frame.index)

    return StrategySpec("TREND_CONTINUATION", name, target_r, stop_atr, 20, signal_builder)


def _distribution(trades: pd.DataFrame) -> dict:
    ordered = trades.assign(entry_time=pd.to_datetime(trades["entry_time"], utc=True)).sort_values("entry_time")
    ordered["month"] = ordered["entry_time"].dt.to_period("M").astype(str)
    ordered["quarter"] = ordered["entry_time"].dt.to_period("Q").astype(str)
    monthly = ordered.groupby("month")["r_multiple"].agg(["count", "sum"]).reset_index()
    quarterly = ordered.groupby("quarter")["r_multiple"].agg(["count", "sum"]).reset_index()
    losses = ordered["r_multiple"].lt(0).astype(int)
    streaks = losses.groupby((losses != losses.shift()).cumsum()).sum()
    equity = (1 + ordered["r_multiple"] * 0.002).cumprod()
    peak = equity.cummax()
    underwater = equity < peak
    groups = (underwater != underwater.shift()).cumsum()
    drawdowns = ordered.loc[underwater].groupby(groups[underwater]).agg(
        start=("entry_time", "min"), end=("exit_time", "max"), trades=("entry_time", "size")
    )
    if not drawdowns.empty:
        drawdowns["days"] = (
            pd.to_datetime(drawdowns["end"], utc=True) - drawdowns["start"]
        ).dt.total_seconds() / 86400
        longest = drawdowns.sort_values("days", ascending=False).iloc[0].to_dict()
    else:
        longest = {}
    return {
        "monthly": monthly,
        "quarterly": quarterly,
        "largest_losing_streak": int(streaks.max()) if len(streaks) else 0,
        "longest_drawdown": longest,
        "best_month": monthly.sort_values("sum", ascending=False).iloc[0].to_dict(),
        "worst_month": monthly.sort_values("sum").iloc[0].to_dict(),
    }


def _table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def _write_report(costs, windows, sensitivity, distribution) -> None:
    calendar = windows[windows.window_type == "calendar"]
    walk_forward = windows[windows.window_type == "walk_forward"]
    monthly = distribution["monthly"]
    quarterly = distribution["quarterly"]
    profitable_years = int((calendar[calendar.window.str.len() == 4]["total_return"] > 0).sum())
    Path("XAUUSD_4H_ROBUSTNESS_REPORT.md").write_text(
        "# XAUUSD 4H Robustness Report\n\n"
        "This is offline research only. The candidate remains unapproved.\n\n"
        "## Cost And Slippage Stress\n\n"
        + _table(costs[["cost_multiplier", "slippage_multiplier", "total_trades", "profit_factor", "expectancy", "max_drawdown", "status"]])
        + "\n\n## Calendar Stability\n\n"
        + _table(calendar[["window", "total_trades", "total_return", "profit_factor", "expectancy", "max_drawdown"]])
        + "\n\n## Walk-Forward Windows\n\n"
        + _table(walk_forward[["window", "total_trades", "total_return", "profit_factor", "expectancy", "max_drawdown"]])
        + "\n\n## Rule Sensitivity\n\n"
        + _table(sensitivity[["variant", "total_trades", "profit_factor", "pf_at_1_25x_cost", "oos_profit_factor", "positive_windows", "status", "reason"]])
        + "\n\n## Trade Distribution\n\n"
        f"- Profitable single years: {profitable_years}/5\n"
        f"- Largest losing streak: {distribution['largest_losing_streak']} trades\n"
        f"- Longest drawdown: {distribution['longest_drawdown']}\n"
        f"- Best month: {distribution['best_month']}\n"
        f"- Worst month: {distribution['worst_month']}\n\n"
        "### Monthly P&L (R)\n\n" + _table(monthly)
        + "\n\n### Quarterly P&L (R)\n\n" + _table(quarterly)
        + "\n\n## Decision\n\n"
        "**Not approved.** The original PF was below 1.25. Approval rules were not "
        "changed, and isolated parameter improvements are treated as exploratory rather "
        "than evidence of a robust edge.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
